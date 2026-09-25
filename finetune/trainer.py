"""Supervised fine-tuning of the laya router on golden-path examples.

Deep-module contract:
    train(bundle, train_examples, dev_examples, cfg) -> TrainResult

Owns: curriculum (short-context stage -> long-context stage), optimizer
groups (encoder vs decision head), the CE objective over option logits,
dev metrics, and final export via laya_backend.save_checkpoint.

Hides: batching/epoch policy, schedules, MPS specifics. Training examples
arrive as dataset.Example dicts already carrying {state, skill, target,
variant}; questions are built here via router_q (the single format
contract) and tokenized via laya_backend.encode_examples (the single
tokenization path).

Context-window goal: stage 2 trains at cfg.max_len_long on long states so
the encoder's RoPE positions beyond the shipped 512-token cap are actually
exercised; ModernBERT natively supports 8192, so this re-adapts pretrained
capacity rather than extending the architecture.
"""

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

import laya_backend as lb
from router_q import build_question

DESC = {"desc"}


@dataclass
class TrainConfig:
    out_dir: Path
    max_len_short: int = 512
    max_len_long: int = 2048
    epochs_short: int = 3
    epochs_long: int = 2
    lr_encoder: float = 1e-5
    lr_head: float = 1e-4
    batch_short: int = 16
    batch_long: int = 4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    seed: int = 13
    smoke: bool = False
    clip_grad_norm: float = 0.0   # 0 disables; per-step MPS clip costs ~5s/step
    max_long_rows: Optional[int] = None  # subsample long-stage rows (stratified)


@dataclass
class TrainResult:
    out_dir: Path
    stage_metrics: dict = field(default_factory=dict)
    final_dev: dict = field(default_factory=dict)


def _example_to_row(ex, skills_by_name: dict) -> dict:
    s = skills_by_name[ex.skill]
    return {
        "state": ex.state,
        "question": build_question(ex.skill, s.get("description", ""), use_desc=(ex.variant in DESC)),
        "target": ex.target,
        "skill": ex.skill,
    }


def _split_examples(examples, skills_by_name):
    """(short_rows, long_rows) keyed by stage; desc/bare variants both kept."""
    short, long = [], []
    for ex in examples:
        if ex.skill not in skills_by_name:
            continue
        row = _example_to_row(ex, skills_by_name)
        (long if ex.state_kind == "long" else short).append(row)
    return short, long


def _loss_and_stats(logits: torch.Tensor, target: torch.Tensor):
    logp = torch.log_softmax(logits.float(), dim=-1)
    loss = -(target * logp).sum(-1).mean()
    with torch.no_grad():
        probs = logp.exp()
        p_core = probs[:, 0]
        is_core = target[:, 0] > 0.5
        is_unrel = target[:, 2] > 0.5
        stats = {
            "loss": float(loss),
            "acc": float((logp.argmax(-1) == target.argmax(-1)).float().mean()),
            "p_core_pos": float(p_core[is_core].mean()) if is_core.any() else float("nan"),
            "p_core_neg": float(p_core[is_unrel].mean()) if is_unrel.any() else float("nan"),
        }
    return loss, stats


@torch.no_grad()
def _eval_items(model, batches, device) -> dict:
    model.eval()
    agg = {"loss": 0.0, "acc": 0.0, "p_core_pos": [], "p_core_neg": [], "n": 0}
    for b in batches:
        logits, _ = model(
            b["input_ids"].to(device), b["attention_mask"].to(device),
            b["marker_pos"].to(device), b["marker_mask"].to(device),
            b["qtype"].to(device),
        )
        _, stats = _loss_and_stats(logits, b["target"].to(device))
        agg["loss"] += stats["loss"] * len(b["label"])
        agg["acc"] += stats["acc"] * len(b["label"])
        if stats["p_core_pos"] == stats["p_core_pos"]:
            agg["p_core_pos"].append(stats["p_core_pos"])
        if stats["p_core_neg"] == stats["p_core_neg"]:
            agg["p_core_neg"].append(stats["p_core_neg"])
        agg["n"] += len(b["label"])
    n = max(agg["n"], 1)
    return {
        "loss": agg["loss"] / n,
        "acc": agg["acc"] / n,
        "p_core_pos": sum(agg["p_core_pos"]) / max(len(agg["p_core_pos"]), 1),
        "p_core_neg": sum(agg["p_core_neg"]) / max(len(agg["p_core_neg"]), 1),
    }


def _cosine_lambda(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * prog))


def run_training(
    bundle: lb.ModelBundle,
    catalog: list[dict],
    train_examples,
    dev_examples,
    cfg: TrainConfig,
    log=None,
) -> TrainResult:
    """The real entry point. `catalog` supplies skill descriptions for the
    desc question variant."""
    if log is None:
        def log(msg):
            print(msg, flush=True)
    skills_by_name = {s["name"]: s for s in catalog}
    short_rows, long_rows = _split_examples(train_examples, skills_by_name)
    dev_short, _ = _split_examples(dev_examples, skills_by_name)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.max_long_rows and len(long_rows) > cfg.max_long_rows:
        # Stratified subsample: preserve the core/tangential/unrelated mix
        # while capping the most expensive stage.
        by_target: dict[str, list] = {}
        for r in long_rows:
            by_target.setdefault(r["target"], []).append(r)
        frac = cfg.max_long_rows / len(long_rows)
        long_rows = [
            r
            for tgt, rows_t in sorted(by_target.items())
            for r in rng.sample(rows_t, max(1, int(len(rows_t) * frac)))
        ]
        log(f"[long] subsampled to {len(long_rows)} rows (stratified x{frac:.2f})")

    if cfg.smoke:
        short_rows, long_rows, dev_short = short_rows[:64], long_rows[:16], dev_short[:32]
        cfg = TrainConfig(**{**cfg.__dict__, "epochs_short": 1, "epochs_long": 1,
                             "max_len_long": min(cfg.max_len_long, 1024),
                             "batch_long": 2, "out_dir": cfg.out_dir})

    device = bundle.agent.device
    model = bundle.agent.model

    # Param groups: decision head moves faster than the encoder; the act head
    # is unused by the router (only p(core) is consumed) and stays frozen.
    head_params, enc_params, frozen = [], [], 0
    for name, p in model.named_parameters():
        if name.startswith("act_head"):
            p.requires_grad_(False)
            frozen += 1
        elif name.startswith(("head", "scorer", "type_emb")):
            head_params.append(p)
        else:
            enc_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cfg.lr_encoder},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    stages = [
        ("short", short_rows, cfg.epochs_short, cfg.batch_short, cfg.max_len_short),
        ("long", long_rows, cfg.epochs_long, cfg.batch_long, cfg.max_len_long),
    ]
    result = TrainResult(out_dir=Path(cfg.out_dir))

    # Freeze the encoder in stage 1 warm-up? No: both stages train all unfrozen
    # params; stage 2 simply uses a lower lr via continued cosine decay handled
    # per-stage below (fresh schedule per stage keeps stage 2 from restarting hot).
    global_step = 0
    # Fresh schedule per stage: carrying global_step across stages squeezed
    # stage 2's lr to zero by its halfway point (the bug that produced a
    # no-op long stage on the first v1 run).
    for stage_name, rows, epochs, batch_n, max_len in stages:
        if not rows:
            log(f"[{stage_name}] no rows, skipping")
            continue
        global_step = 0
        items = lb.encode_examples(bundle, rows, max_len=max_len)
        # Quantize lengths to SHAPE_BAND tokens: consecutive batches then share
        # identical padded shapes, which MPS kernel/allocator caches reward
        # (~3x). Sort by band, not raw length — monotonic raw lengths give
        # every batch a brand-new shape, the slowest possible pattern.
        band = 64
        for it in items:
            target = min(max_len, -(-len(it["ids"]) // band) * band)
            it["ids"] = it["ids"] + [bundle.agent.tok.pad_token_id] * (target - len(it["ids"]))
        items.sort(key=lambda it: len(it["ids"]))
        dev_items = lb.encode_examples(bundle, dev_short, max_len=min(max_len, cfg.max_len_short)) if stage_name == "short" else []
        dev_batches = []
        for i in range(0, len(dev_items), batch_n * 2):
            b = lb.collate_chunk(dev_items[i : i + batch_n * 2], bundle.agent.tok.pad_token_id)
            if b:
                dev_batches.append(b)

        n_batches = math.ceil(len(items) / batch_n)
        total_steps = n_batches * epochs
        warmup = max(int(total_steps * cfg.warmup_frac), 2)
        base_lrs = [g["lr"] for g in opt.param_groups]

        log(f"[{stage_name}] {len(items)} rows, {n_batches} batches x {epochs} epochs, max_len={max_len}")
        for epoch in range(epochs):
            order = list(range(n_batches))
            rng.shuffle(order)
            model.train()
            ep = {"loss": 0.0, "acc": 0.0, "n": 0, "steps": 0}
            t0 = time.time()
            for bi in order:
                chunk = items[bi * batch_n : (bi + 1) * batch_n]
                b = lb.collate_chunk(chunk, bundle.agent.tok.pad_token_id)
                if b is None:
                    continue
                scale = _cosine_lambda(global_step, total_steps, warmup)
                for g, base in zip(opt.param_groups, base_lrs):
                    g["lr"] = base * scale
                opt.zero_grad()
                logits, _ = model(
                    b["input_ids"].to(device), b["attention_mask"].to(device),
                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                    b["qtype"].to(device),
                )
                loss, stats = _loss_and_stats(logits, b["target"].to(device))
                loss.backward()
                if cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in opt.param_groups for p in g["params"]], cfg.clip_grad_norm
                    )
                opt.step()
                ep["loss"] += stats["loss"]; ep["acc"] += stats["acc"]
                ep["n"] += 1; ep["steps"] += 1
                global_step += 1
                if ep["steps"] % 20 == 0:
                    log(f"  [{stage_name} e{epoch} step {ep['steps']}/{n_batches}] "
                        f"loss={stats['loss']:.4f} acc={stats['acc']:.3f} "
                        f"lr_scale={scale:.2f} ({time.time()-t0:.0f}s)")
            dev = _eval_items(model, dev_batches, device) if dev_batches else {}
            log(f"  [{stage_name} epoch {epoch}] train loss={ep['loss']/max(ep['n'],1):.4f} "
                f"acc={ep['acc']/max(ep['n'],1):.3f} "
                + (f"dev loss={dev['loss']:.4f} acc={dev['acc']:.3f} "
                   f"p_core pos/neg={dev['p_core_pos']:.3f}/{dev['p_core_neg']:.3f}" if dev else ""))
            result.stage_metrics[f"{stage_name}_e{epoch}"] = {
                "train_loss": ep["loss"] / max(ep["n"], 1), "train_acc": ep["acc"] / max(ep["n"], 1), **dev
            }

    # Serve-time config keeps the long context that stage 2 trained in; the
    # eval ablation quantifies the latency cost before this ships anywhere.
    serve_max_len = cfg.max_len_long if long_rows else cfg.max_len_short
    note = "smoke" if cfg.smoke else f"SFT golden-path; stages short@{cfg.max_len_short}, long@{cfg.max_len_long}"
    out = lb.save_checkpoint(bundle, cfg.out_dir, max_len=serve_max_len, note=note)
    result.out_dir = out
    result.final_dev = result.stage_metrics
    log(f"[export] {out} (serve max_len={serve_max_len})")
    return result
