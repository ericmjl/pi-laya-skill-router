"""Offline eval: score checkpoints against held-out turns.

Deep-module contract:
    evaluate(ckpt, turns, golden, skills, ...) -> EvalReport
    reports_to_md({name: report}) -> str

References: every labeled turn is scored against BOTH ground truths —
  observed  skills actually read via `read` (comparable to eval/RESULTS.md)
  golden    frontier-model labels (loaded + should_have) — the direct
            measure of "selection matched against a frontier model"
plus no-load turn behavior (the probability-crowding failure mode),
suppression of the router's own mined false positives (fp_suppression_rate —
None when no picks caches are passed), and latency.
`max_len_override` enables the context-window ablation on a single
checkpoint without touching its config.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import laya_backend as lb
from dataset import classify_pick, turn_truth
from session_data import Turn, session_split

REFS = ("observed", "golden")
KS = (1, 3, 5, 10)


@dataclass
class EvalReport:
    name: str
    n_turns: int = 0
    n_labeled: int = 0
    n_noload: int = 0
    recall: dict = field(default_factory=dict)      # (ref, k) -> mean
    precision_at_3: float = 0.0
    gold_rank_median: Optional[int] = None
    noload_picks_mean: float = 0.0
    noload_picks_at_k: float = 0.0                  # picks within top-k on no-load turns
    fp_suppression_rate: Optional[float] = None     # router's own FPs now below threshold
    n_fp_turns: int = 0                             # test turns with ≥1 known FP pick
    latency_p50_ms: int = 0
    latency_max_ms: int = 0
    max_len: int = 0


def _state_for(turn: Turn, kind: str) -> str:
    if kind == "long":
        return turn.long_state()
    state = turn.state
    if len(state.strip()) <= 60 and turn.prev_state:
        return (turn.prev_state + " || " + state).strip()
    return state


def evaluate(
    name: str,
    ckpt_id: str,
    turns: list[Turn],
    golden: dict,
    skills: list[dict],
    picks: Optional[dict] = None,
    split: str = "test",
    use_desc: bool = False,
    state_kind: str = "short",
    threshold: float = 0.3,
    top_k: int = 10,
    max_len_override: Optional[int] = None,
    limit: Optional[int] = None,
    log=print,
) -> EvalReport:
    catalog = {s["name"] for s in skills}
    test_turns = [t for t in turns if session_split(t.session) == split]
    if limit:
        test_turns = test_turns[:limit]
    bundle = lb.load_checkpoint(ckpt_id)
    max_len = max_len_override or bundle.max_len

    rep = EvalReport(name=name, max_len=max_len)
    rec_hits = {r: {k: [] for k in KS} for r in REFS}
    gold_ranks: list[int] = []
    prec3: list[float] = []
    noload: list[int] = []
    noload_k: list[int] = []
    fp_below: list[int] = []   # per FP pick: 1 if now below threshold
    latencies: list[int] = []

    for i, turn in enumerate(test_turns):
        truth = turn_truth(turn, golden.get(turn.session, {}).get(turn.turn_index), catalog)
        if not truth.known:
            continue
        rep.n_turns += 1
        # The router's own picks on this turn (mine_picks.py), classified
        # against the same merge the dataset trains on — over-picked skills
        # the truth says were neither needed nor tangential.
        fp_picks = [v.skill for v in (picks or {}).get(turn.key, [])
                    if classify_pick(truth, v.skill) == "unrelated"]
        ranking = lb.predict_ranking(
            bundle, _state_for(turn, state_kind), skills,
            use_desc=use_desc, max_len=max_len, threshold=threshold, top_k=top_k,
        )
        latencies.append(ranking.latency_ms)
        ranked_names = [n for n, _ in ranking.ranked]
        pick_names = [n for n, _ in ranking.picks]

        if truth.positives:
            rep.n_labeled += 1
            golden_pos = set(truth.positives)
            observed_pos = {s for s in turn.observed_labels if s in catalog}
            for ref, pos in (("observed", observed_pos), ("golden", golden_pos)):
                if not pos:
                    continue
                for k in KS:
                    hits = len(pos & set(ranked_names[:k]))
                    rec_hits[ref][k].append(hits / len(pos))
            ranks = [ranked_names.index(s) + 1 for s in golden_pos if s in ranked_names]
            gold_ranks.extend(ranks)
            prec3.append(len(set(pick_names[:3]) & golden_pos) / 3)
        else:
            rep.n_noload += 1
            noload.append(len(pick_names))
            noload_k.append(len(ranked_names[:top_k]))
        if fp_picks:
            rep.n_fp_turns += 1
            scores = dict(ranking.ranked)
            fp_below.extend(1 if scores.get(s, 0.0) < threshold else 0 for s in fp_picks)
        if (i + 1) % 25 == 0:
            log(f"  eval {i + 1}/{len(test_turns)}")

    for ref in REFS:
        for k in KS:
            vals = rec_hits[ref][k]
            if vals:
                rep.recall[f"{ref}@{k}"] = sum(vals) / len(vals)
    rep.precision_at_3 = sum(prec3) / len(prec3) if prec3 else 0.0
    rep.gold_rank_median = sorted(gold_ranks)[len(gold_ranks) // 2] if gold_ranks else None
    rep.noload_picks_mean = sum(noload) / len(noload) if noload else 0.0
    rep.noload_picks_at_k = sum(noload_k) / len(noload_k) if noload_k else 0.0
    if fp_below:
        rep.fp_suppression_rate = sum(fp_below) / len(fp_below)
    if latencies:
        s = sorted(latencies)
        rep.latency_p50_ms = s[len(s) // 2]
        rep.latency_max_ms = s[-1]
    return rep


def reports_to_md(reports: dict[str, EvalReport]) -> str:
    head = (
        "| checkpoint | max_len | turns | labeled | recall@1 | recall@3 | recall@5 | recall@10"
        " | golden-recall@3 | prec@3 | gold-rank med | no-load picks | fp-supp | p50 ms | max ms |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for rep in reports.values():
        r = lambda key: f"{rep.recall.get(key, 0.0):.3f}" if rep.recall.get(key) is not None else "-"
        fp = f"{rep.fp_suppression_rate:.3f} ({rep.n_fp_turns})" if rep.fp_suppression_rate is not None else "-"
        rows.append(
            f"| {rep.name} | {rep.max_len} | {rep.n_turns} | {rep.n_labeled} "
            f"| {r('observed@1')} | {r('observed@3')} | {r('observed@5')} | {r('observed@10')} "
            f"| {r('golden@3')} | {rep.precision_at_3:.3f} "
            f"| {rep.gold_rank_median if rep.gold_rank_median else '-'} "
            f"| {rep.noload_picks_mean:.1f} | {fp} "
            f"| {rep.latency_p50_ms} | {rep.latency_max_ms} |"
        )
    return head + "\n".join(rows) + "\n"


def main() -> None:
    import argparse

    from dataset import load_golden
    from session_data import load_turns

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="checkpoint name=path (repeatable). name=baseline path=convaiinnovations/laya")
    ap.add_argument("--split", default="test")
    ap.add_argument("--use-desc", action="store_true")
    ap.add_argument("--state-kind", default="short", choices=("short", "long"))
    ap.add_argument("--max-len", default="", help="comma list: ablation on the LAST --ckpt")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--picks-dir", type=Path, default=repo / "finetune" / "picks",
                    help="mine_picks cache dir; empty/missing disables the fp metric")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="eval/finetune_results.md")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    skills = json.load(open(repo / "skills.json"))
    turns = load_turns()
    golden = load_golden(repo / "finetune" / "distilled")
    from mine_picks import load_picks
    picks = load_picks(args.picks_dir)

    ckpts = []
    for spec in args.ckpt or ["baseline=convaiinnovations/laya"]:
        name, _, path = spec.partition("=")
        ckpts.append((name or "baseline", path or "convaiinnovations/laya"))

    reports = {}
    ablate = [int(x) for x in args.max_len.split(",") if x.strip()]
    for name, path in ckpts:
        if ablate and (name, path) == ckpts[-1]:
            for ml in ablate:
                reports[f"{name}@len{ml}"] = evaluate(
                    f"{name}@len{ml}", path, turns, golden, skills, picks=picks, split=args.split,
                    use_desc=args.use_desc, state_kind=args.state_kind, threshold=args.threshold,
                    top_k=args.top_k, max_len_override=ml, limit=args.limit)
        else:
            reports[name] = evaluate(
                name, path, turns, golden, skills, picks=picks, split=args.split,
                use_desc=args.use_desc, state_kind=args.state_kind, threshold=args.threshold,
                top_k=args.top_k, limit=args.limit)

    md = reports_to_md(reports)
    print(md)
    Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
