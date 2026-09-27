"""Distill the laya skill-router cross-encoder into a dual-tower bi-encoder.

Why: the cross-encoder scores one (state, skill) row per skill — ~19.5ms/row
on MPS, ~2.7s for the full catalog. A bi-encoder embeds the state once and
ranks it against cached skill-document vectors, which is the only way to
shortlist without giving up the hard matches (lexical and zero-shot dense
stage-1 both measured below the full-catalog bar — see
docs/ROUTER_LATENCY_STRATEGY.md, dead ends 3-5). The shortlist it produces
feeds the same laya cross-encoder over ~16 rows instead of ~110, so the
calibrated scorer — and therefore pick quality — is unchanged as long as the
gold skill is inside the shortlist. That recall is the number this script
optimizes and reports.

Teacher: `laya_backend.predict_ranking` through the fine-tuned checkpoint
(default finetune/checkpoints/v1) — the exact production scoring path.
Labels: `dataset.turn_truth`, the single source of truth in the repo.

Losses:
  - InfoNCE over the full catalog (every skill is a candidate, matching the
    serve-time task; multi-positive turns average over positives)
  - distribution match (MSE between softmaxed teacher p_core and student
    cosine) — transfers the cross-encoder's ranking, not just its argmax

Promotion gate (run after training, before pointing LAYA_BIENCODER at it):
  1. test-split shortlist recall@k printed by this script must not sit below
     full-catalog recall@3 from eval/FINETUNE_RESULTS.md minus noise
  2. `LAYA_BIENCODER=<out> LAYA_SHORTLIST_K=16 uv run python sidecar/server.py`
     then `uv run scripts/run_eval.py` must match the full-catalog run
     within noise on recall@1/3/5, precision@3, and no-load picks

Usage:
  uv run python finetune/train_biencoder.py                # full run
  uv run python finetune/train_biencoder.py --max-states 40  # smoke
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

os_default = None
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "finetune"))

from dataset import load_golden, turn_truth  # noqa: E402
from laya_backend import load_checkpoint, predict_ranking  # noqa: E402
from session_data import load_turns, session_split  # noqa: E402

MIN_SHORT_STATE = 60  # same prev-turn-prepend rule as run_eval v2 / dataset.py
DOC_CAP = 400         # skill-description cap mirrored by the sidecar's stage-1
STATE_CAP = 700       # production state cap (sidecar LAYA_STATE_MAX_CHARS)


def route_state(turn) -> str:
    if len(turn.state) >= MIN_SHORT_STATE:
        return turn.state
    return f"{turn.prev_state} || {turn.state}".strip()


def effective_state(turn) -> str:
    return route_state(turn)[:STATE_CAP]


def load_catalog(skills_path: Path) -> list[dict]:
    skills = json.loads(skills_path.read_text())
    return [{"name": s["name"], "description": s.get("description", "")} for s in skills]


def doc_text(skill: dict) -> str:
    return f"{skill['name']}: {skill['description'][:DOC_CAP]}"


# --------------------------------------------------------------------------- teacher


def teacher_scores(
    bundle, states: list[str], cache_path: Path, catalog: list[dict],
    max_len: int | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """state-hash -> [(skill, p_core)] ranked desc, through the real predict path."""
    cache: dict[str, list[tuple[str, float]]] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                row = json.loads(line)
                cache[row["h"]] = [(n, p) for n, p in row["ranked"]]
            except (json.JSONDecodeError, KeyError):
                continue
    todo = sorted({s for s in states if _sh(s) not in cache})  # dedupe repeated states
    print(f"teacher scores: {len(cache)} cached, {len(todo)} to compute "
          f"(~{len(todo) * 2.8 / 60:.0f} min on MPS)", flush=True)
    if todo:
        skills_payload = [{"name": s["name"], "description": s.get("description", "")} for s in catalog]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a") as fh:
            for i, state in enumerate(todo):
                t0 = time.time()
                r = predict_ranking(bundle, state, skills_payload, threshold=0.0, top_k=1, max_len=max_len)
                cache[_sh(state)] = r.ranked
                fh.write(json.dumps({"h": _sh(state), "ranked": r.ranked}) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(todo)} ({(time.time() - t0):.1f}s last)", flush=True)
    return cache


def _sh(state: str) -> str:
    return hashlib.sha1(state.encode()).hexdigest()


# --------------------------------------------------------------------------- model


class BiEncoder(torch.nn.Module):
    """Shared-weight dual tower over the checkpoint's ModernBERT, mean-pooled.

    The encoder module comes from `load_checkpoint` (the laya checkpoint
    format stores the full DecisionModel state dict at the top level — an
    `AutoModel.from_pretrained(checkpoint/encoder)` load would silently
    build an untrained encoder from config alone).
    """

    def __init__(self, encoder: torch.nn.Module, tok, device: str):
        super().__init__()
        self.encoder = encoder.to(device)
        self.tok = tok
        self.device = device

    def _pool(self, texts: list[str], max_length: int) -> torch.Tensor:
        # No no_grad here: training backprops through both towers. Eval call
        # sites wrap their own torch.no_grad().
        enc = self.tok(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self.device)
        out = self.encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (out * mask).sum(1) / mask.sum(1)
        return F.normalize(vec, dim=-1)

    def embed_states(self, states: list[str]) -> torch.Tensor:
        return self._pool(states, 256)

    def embed_docs(self, docs: list[str]) -> torch.Tensor:
        return self._pool(docs, 128)


def train(
    model: BiEncoder,
    train_states: list[str],
    train_truth: dict[str, set[str]],
    train_teacher: dict[str, list[tuple[str, float]]],
    catalog: list[dict],
    dev: list[tuple[str, set[str]]],
    epochs: int,
    batch_size: int,
    lr: float,
    tau_nce: float,
    tau_dist: float,
    distill_weight: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    names = [s["name"] for s in catalog]
    docs = [doc_text(s) for s in catalog]
    name_to_idx = {n: i for i, n in enumerate(names)}

    def ks_at(ranks: torch.Tensor, gold: list[set[str]], ks=(8, 16, 24)) -> dict[int, float]:
        hits = {k: 0 for k in ks}
        for i, g in enumerate(gold):
            if not g:
                continue
            top = ranks[i].argsort(descending=True)[: max(ks)].tolist()
            for k in ks:
                if any(names[j] in g for j in top[:k]):
                    hits[k] += 1
        n = sum(1 for g in gold if g)
        return {k: hits[k] / n for k in ks}

    for epoch in range(epochs):
        model.train()
        order = list(train_states)
        rng.shuffle(order)
        total_loss = 0.0
        nb = 0
        for i in range(0, len(order), batch_size):
            batch = order[i : i + batch_size]
            # Doc embeddings recompute per batch: a per-epoch tensor would be
            # backwarded through once, then freed — the second batch crashes
            # trying to reuse the graph. retain_graph would pile the whole
            # epoch into one autograd graph instead.
            doc_vecs = model.embed_docs(docs)                    # (n_skills, d)
            sv = model.embed_states(batch)                       # (b, d)
            sims = sv @ doc_vecs.T                               # (b, n_skills)

            loss = sims.new_tensor(0.0)
            for j, state in enumerate(batch):
                teacher = train_teacher.get(_sh(state))
                gold = train_truth.get(state, set())
                # InfoNCE over the full catalog with soft positives
                if gold:
                    pos_idx = [name_to_idx[g] for g in gold if g in name_to_idx]
                    if pos_idx:
                        logp = F.log_softmax(sims[j] / tau_nce, dim=-1)
                        loss = loss - torch.stack([logp[p] for p in pos_idx]).mean()
                # Distribution match with the teacher over the same softmax
                if teacher:
                    tp = torch.tensor([p for _, p in teacher],
                                      device=sims.device, dtype=sims.dtype)
                    tp = F.softmax(tp / tau_dist, dim=-1)
                    sp = F.log_softmax(sims[j] / tau_dist, dim=-1).exp()
                    loss = loss + distill_weight * F.mse_loss(sp, tp)
            loss = loss / max(1, len(batch))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            nb += 1
        model.eval()
        with torch.no_grad():
            dv = model.embed_docs(docs)
            dev_states = [s for s, _ in dev]
            dev_gold = [g for _, g in dev]
            if dev_states:
                q = model.embed_states(dev_states)
                ranks = q @ dv.T
                rec = ks_at(ranks, dev_gold)
                print(f"epoch {epoch}: loss {total_loss / max(1, nb):.4f} | dev shortlist "
                      f"recall@8 {rec[8]:.3f} @16 {rec[16]:.3f} @24 {rec[24]:.3f}", flush=True)


@torch.no_grad()
def eval_split(model: BiEncoder, catalog: list[dict], rows: list[tuple[str, set[str]]], tag: str) -> dict:
    model.eval()
    names = [s["name"] for s in catalog]
    docs = [doc_text(s) for s in catalog]
    dv = model.embed_docs(docs)
    q = model.embed_states([s for s, _ in rows])
    ranks = (q @ dv.T).argsort(descending=True, dim=-1)
    ks = (1, 3, 5, 8, 16, 24)
    hits = {k: 0 for k in ks}
    mrr = 0.0
    n = 0
    for i, (_, g) in enumerate(rows):
        if not g:
            continue
        n += 1
        top = ranks[i].tolist()
        rank_of = {names[j]: r for r, j in enumerate(top)}
        best = min((rank_of[s] for s in g if s in rank_of), default=None)
        if best is not None:
            mrr += 1.0 / (best + 1)
        for k in ks:
            if any(names[j] in g for j in top[:k]):
                hits[k] += 1
    out = {f"recall@{k}": hits[k] / max(1, n) for k in ks}
    out["mrr"] = mrr / max(1, n)
    print(f"{tag}: n={n} | " + " ".join(f"{a} {b:.3f}" for a, b in out.items()), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    _default_ckpt = Path.home() / ".pi" / "agent" / "laya-router" / "checkpoints" / "v1"
    if not _default_ckpt.exists():
        _default_ckpt = REPO / "finetune" / "checkpoints" / "v1"
    ap.add_argument("--checkpoint", default=str(_default_ckpt))
    ap.add_argument("--out", default=str(REPO / "finetune" / "checkpoints" / "biencoder-v1"))
    ap.add_argument("--skills", default=str(REPO / "skills.json"))
    ap.add_argument("--golden-dir", default=str(REPO / "finetune" / "distilled"))
    ap.add_argument("--teacher-cache", default=str(REPO / "finetune" / "cache" / "teacher_scores_v1.jsonl"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tau-nce", type=float, default=0.07)
    ap.add_argument("--tau-dist", type=float, default=0.1)
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-states", type=int, default=0, help="debug cap on turns")
    args = ap.parse_args()

    catalog = load_catalog(Path(args.skills))
    catalog_names = {s["name"] for s in catalog}
    print(f"catalog: {len(catalog)} skills", flush=True)

    turns = load_turns()
    golden = load_golden(Path(args.golden_dir))
    rows: list[tuple[object, set[str]]] = []  # (turn, gold set)
    skipped = 0
    for t in turns:
        g = golden.get(t.session, {}).get(t.turn_index)
        truth = turn_truth(t, g, catalog_names)
        if not truth.known:
            skipped += 1  # same policy as build_dataset: unknown truth never trains
            continue
        rows.append((t, set(truth.positives.keys())))
    if args.max_states:
        rows = rows[: args.max_states]
    print(f"turns: {len(rows)} with signal ({skipped} unknown skipped)", flush=True)

    states = [effective_state(t) for t, _ in rows]
    truth_by_state = {effective_state(t): g for t, g in rows if g}

    print("loading teacher (cross-encoder) ...", flush=True)
    bundle = load_checkpoint(args.checkpoint, device=args.device)
    teacher = teacher_scores(bundle, states, Path(args.teacher_cache), catalog)

    # session-level split: no leakage between train and eval turns
    train_rows = [(t, g) for t, g in rows if session_split(t.session) == "train"]
    dev_rows = [(effective_state(t), g) for t, g in rows
                if session_split(t.session) in ("dev", "test") and g]
    train_states = sorted({effective_state(t) for t, _ in train_rows})
    print(f"train states: {len(train_states)}, dev+test gold turns: {len(dev_rows)}", flush=True)

    model = BiEncoder(bundle.agent.model.encoder, bundle.agent.tok, args.device)
    train(
        model,
        train_states,
        truth_by_state,
        teacher,
        catalog,
        dev_rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau_nce=args.tau_nce,
        tau_dist=args.tau_dist,
        distill_weight=args.distill_weight,
        seed=args.seed,
    )

    print("\n--- held-out shortlist recall (promotion gate input) ---", flush=True)
    eval_split(model, catalog, dev_rows, "dev+test")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(out / "encoder"))
    # The tokenizer is untouched by training; copy the checkpoint's own
    # tokenizer directory verbatim. save_pretrained writes a "TokenizersBackend"
    # class tag that AutoTokenizer in this venv cannot load back.
    shutil.copytree(Path(args.checkpoint) / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    (out / "biencoder_config.json").write_text(json.dumps({
        "distilled_from": args.checkpoint,
        "pooling": "mean",
        "state_max_chars": STATE_CAP,
        "doc_max_chars": DOC_CAP,
        "tau_nce": args.tau_nce,
        "tau_dist": args.tau_dist,
        "seed": args.seed,
    }, indent=2))
    print(f"\nsaved bi-encoder to {out}", flush=True)
    print("promotion gate:", flush=True)
    print(f"  1. compare the recall@k table above against full-catalog recall@3 in eval/FINETUNE_RESULTS.md", flush=True)
    print(f"  2. LAYA_BIENCODER={out} LAYA_SHORTLIST_K=16 uv run python sidecar/server.py", flush=True)
    print(f"  3. uv run scripts/run_eval.py  # must match the full-catalog run within noise", flush=True)


if __name__ == "__main__":
    main()
