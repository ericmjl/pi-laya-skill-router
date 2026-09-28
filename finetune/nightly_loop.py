"""Nightly auto-learning loop: distill new sessions, retrain, promote if better.

Cadence: launchd runs this nightly (see launchd/com.ericmjl.laya-nightly.plist).
Each run:

    1. distill   frontier-model golden labels for NEW sessions only
                 (per-session caches make this incremental; capped per run
                 to bound API cost; labeler model pinned for label consistency)
    2. train     retrain FROM BASE on the full ever-expanding golden set —
                 never from last night's checkpoint, so errors don't compound
    3. eval      candidate AND production on the current test split; the
                 session-hash split is deterministic, so the test holdout
                 grows with the golden set but never leaks into training
    4. gate      promote only if the candidate beats production on
                 golden-recall@3, does not regress no-load picks, and stays
                 inside the latency budget; otherwise the run is recorded
                 and discarded
    5. record    append history.jsonl, write a per-run report, commit the
                 expanded golden set + history to git

Failure at any stage never touches the live sidecar: promotion is the only
step that interacts with it, and it runs last with its own health-check
rollback.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_golden
from distiller import DEFAULT_MODEL, distill_turns
from eval_ckpt import EvalReport, evaluate, reports_to_md
from laya_backend import DEFAULT_BASE
from mine_picks import MineReport, mine_picks
from session_data import DEFAULT_SESSIONS_DIR, load_turns

REPO = Path(__file__).resolve().parent.parent
DURABLE_CKPT = Path.home() / ".pi" / "agent" / "laya-router" / "checkpoints" / "v1"
ROUTE_LOG = Path.home() / ".pi" / "agent" / "laya-router" / "log.jsonl"


@dataclass
class LoopConfig:
    distill_cap: int = 40              # max NEW sessions labeled per night
    distill_model: str = DEFAULT_MODEL
    distill_workers: int = 4
    max_long_rows: int = 1200
    epochs_long: int = 1
    time_budget_min: int = 110         # hard wall-clock budget for stages 1-4
    promote: bool = True
    keep_checkpoints: int = 3
    smoke: bool = False
    # promotion gate
    min_golden_recall3: float = 0.0    # candidate must be strictly above production
    max_noload_regression: float = 0.5  # candidate picks may not exceed prod + this
    max_fp_regression: float = 0.02    # candidate may not suppress the router's own FPs much worse than prod
    max_latency_p50_ms: int = 4000


@dataclass
class Decision:
    promote: bool
    reasons: list[str] = field(default_factory=list)


def decide_promotion(cand: EvalReport, prod: EvalReport, cfg: LoopConfig) -> Decision:
    """The promotion gate. Pure function: metrics in, decision + reasons out."""
    reasons = []
    cand_recall = cand.recall.get("golden@3", 0.0)
    prod_recall = prod.recall.get("golden@3", 0.0)
    if cand_recall > prod_recall + cfg.min_golden_recall3:
        reasons.append(f"golden-recall@3 {prod_recall:.3f} -> {cand_recall:.3f}")
    else:
        return Decision(False, [f"golden-recall@3 not improved: {prod_recall:.3f} -> {cand_recall:.3f}"])
    if cand.noload_picks_mean > prod.noload_picks_mean + cfg.max_noload_regression:
        return Decision(False, reasons + [
            f"no-load picks regressed: {prod.noload_picks_mean:.1f} -> {cand.noload_picks_mean:.1f}"])
    reasons.append(f"no-load picks {prod.noload_picks_mean:.1f} -> {cand.noload_picks_mean:.1f}")
    if prod.fp_suppression_rate is not None and cand.fp_suppression_rate is not None:
        if cand.fp_suppression_rate < prod.fp_suppression_rate - cfg.max_fp_regression:
            return Decision(False, reasons + [
                f"fp suppression regressed: {prod.fp_suppression_rate:.3f} -> {cand.fp_suppression_rate:.3f} "
                f"(allowed -{cfg.max_fp_regression})"])
        reasons.append(f"fp suppression {prod.fp_suppression_rate:.3f} -> {cand.fp_suppression_rate:.3f}")
    if cand.latency_p50_ms > cfg.max_latency_p50_ms:
        return Decision(False, reasons + [f"latency p50 {cand.latency_p50_ms}ms over budget"])
    reasons.append(f"latency p50 {cand.latency_p50_ms}ms within budget")
    return Decision(True, reasons)


def _new_sessions(turns, distilled_dir: Path, cap: int):
    """Turns belonging to sessions that have no distiller cache yet, capped."""
    have = {p.stem for p in distilled_dir.glob("*.json")}
    by_session = {}
    for t in turns:
        by_session.setdefault(t.session, []).append(t)
    fresh = [s for s in sorted(by_session) if s not in have][:cap]
    return [t for s in fresh for t in by_session[s]], len(fresh)


def _distill(cfg: LoopConfig, log) -> int:
    skills = json.load(open(REPO / "skills.json"))
    turns = load_turns()
    distilled = REPO / "finetune" / "distilled"
    fresh_turns, n_new = _new_sessions(turns, distilled, cfg.distill_cap)
    if not fresh_turns:
        log("distill: no new sessions")
        return 0
    log(f"distill: {n_new} new sessions ({len(fresh_turns)} turns), model={cfg.distill_model}")
    _, report = distill_turns(
        fresh_turns, skills, out_dir=distilled, model=cfg.distill_model,
        workers=cfg.distill_workers, force=False,
    )
    log(report.summary())
    return len(fresh_turns)


def _mine(cfg: LoopConfig, log) -> tuple[dict, MineReport]:
    """Join the route log's picks to their turns (cheap, log-local).
    Caches rewrite only on content change, so sessions_updated is the
    there-is-new-signal signal for the loop."""
    turns = load_turns()
    picks, rep = mine_picks(turns, ROUTE_LOG, REPO / "finetune" / "picks")
    log("mine: " + rep.summary())
    return picks, rep


def _train(cfg: LoopConfig, out: Path, log) -> dict:
    import laya_backend as lb
    from dataset import load_examples
    from trainer import TrainConfig, run_training

    skills = json.load(open(REPO / "skills.json"))
    data = REPO / "finetune" / "data"
    # rebuild from the full golden set
    import subprocess as sp
    r = sp.run([sys.executable, str(REPO / "finetune" / "build_dataset.py")],
               capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"build_dataset failed: {r.stderr[-500:]}")
    log(r.stdout.strip().splitlines()[-1])

    train = load_examples(data / "train.jsonl")
    dev = load_examples(data / "dev.jsonl")
    bundle = lb.load_checkpoint(DEFAULT_BASE)
    tcfg = TrainConfig(
        out_dir=out, max_long_rows=cfg.max_long_rows, epochs_long=cfg.epochs_long,
        smoke=cfg.smoke,
    )
    result = run_training(bundle, skills, train, dev, tcfg, log=log)
    return result.stage_metrics


def _eval_pair(cfg: LoopConfig, cand_dir: Path, picks: dict, log) -> tuple[EvalReport, EvalReport]:
    skills = json.load(open(REPO / "skills.json"))
    turns = load_turns()
    golden = load_golden(REPO / "finetune" / "distilled")
    limit = 30 if cfg.smoke else None
    log("eval: candidate on test split")
    cand = evaluate("candidate", str(cand_dir), turns, golden, skills, picks=picks,
                    split="test", limit=limit, log=log)
    log("eval: production on test split")
    prod = evaluate("production", str(DURABLE_CKPT), turns, golden, skills, picks=picks,
                    split="test", limit=limit, log=log)
    return cand, prod


def _promote(cand_dir: Path, log) -> None:
    if DURABLE_CKPT.exists():
        backup = DURABLE_CKPT.parent / "v1-prev"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(DURABLE_CKPT, backup)
        log(f"promote: backed up current to {backup.name}")
    subprocess.run(["rsync", "-a", "--delete", f"{cand_dir}/", f"{DURABLE_CKPT}/"], check=True)
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/com.ericmjl.laya-sidecar"], check=True)
    import urllib.request
    for _ in range(20):
        time.sleep(3)
        try:
            with urllib.request.urlopen("http://127.0.0.1:7699/health", timeout=5) as r:
                h = json.load(r)
                if h.get("loaded"):
                    log(f"promote: sidecar healthy on {h.get('model')}")
                    return
        except Exception:
            pass
    # health never came back: roll back
    subprocess.run(["rsync", "-a", "--delete",
                    f"{DURABLE_CKPT.parent / 'v1-prev'}/", f"{DURABLE_CKPT}/"], check=True)
    subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/com.ericmjl.laya-sidecar"], check=True)
    raise RuntimeError("promotion health check failed; rolled back")


def _prune(keep: int, log) -> None:
    runs = sorted((REPO / "finetune" / "checkpoints").glob("????-??-??"))
    for old in runs[:-keep] if len(runs) > keep else []:
        shutil.rmtree(old, ignore_errors=True)
        log(f"pruned {old.name}")


def _record(cfg: LoopConfig, run_dir: Path, cand: EvalReport, prod: EvalReport,
            decision: Decision, n_new_sessions: int, mine_rep: MineReport | None,
            status: str, minutes: float) -> Path:
    loop_dir = REPO / "finetune" / "loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date.today().isoformat(),
        "status": status,
        "new_sessions": n_new_sessions,
        "mined_picks": asdict(mine_rep) if mine_rep else None,
        "minutes": round(minutes, 1),
        "decision": asdict(decision),
        "candidate": asdict(cand),
        "production": asdict(prod),
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=1))
    (run_dir / "report.md").write_text(
        f"# nightly run {payload['date']} — {status}\n\n"
        f"new sessions distilled: {n_new_sessions}; wall: {minutes:.0f} min\n\n"
        + reports_to_md({"candidate": cand, "production": prod}) + "\n## gate\n\n"
        + "\n".join(f"- {r}" for r in decision.reasons) + "\n"
    )
    with (loop_dir / "history.jsonl").open("a") as f:
        f.write(json.dumps(payload) + "\n")
    return run_dir


def run_nightly(cfg: LoopConfig, log=None) -> int:
    log = log or (lambda m: print(m, flush=True))
    t0 = time.time()
    day = date.today().isoformat()
    cand_dir = REPO / "finetune" / "checkpoints" / ("smoke" if cfg.smoke else day)
    status, cand, prod, decision, n_new = "failed", None, None, Decision(False, []), 0
    mine_rep: MineReport | None = None

    def budget_left() -> bool:
        return (time.time() - t0) < cfg.time_budget_min * 60

    try:
        if not (REPO / "skills.json").exists():
            subprocess.run([sys.executable, str(REPO / "scripts" / "scan_skills.py")], check=True)
        n_new = _distill(cfg, log) if budget_left() else 0

        # Mine the router's own picks off the route log (cheap, no model).
        # Runs even with zero new sessions: the first run after deploying the
        # session-id-logging extension has a backlog of picks to join, and
        # those FPs alone are new training signal worth a retrain.
        picks: dict = {}
        if budget_left():
            picks, mine_rep = _mine(cfg, log)

        if n_new == 0 and not cfg.smoke:
            if mine_rep is None or mine_rep.sessions_updated == 0:
                status = "no-new-data"
                log("no new sessions; nothing to retrain on")
                return 0
            log("no new sessions, but newly mined picks; retraining on them")

        cand = None
        if budget_left():
            metrics = _train(cfg, cand_dir, log)
            if cfg.smoke:
                status = "smoke"
        else:
            status = "out-of-budget-pre-train"
            log("out of time budget before training; recording only")

        if cand_dir.exists() and budget_left():
            cand, prod = _eval_pair(cfg, cand_dir, picks, log)
        if cand is not None and prod is not None:
            decision = decide_promotion(cand, prod, cfg)
            log("gate: " + ("; ".join(decision.reasons)))
            if decision.promote and cfg.promote:
                _promote(cand_dir, log)
                status = "promoted"
            elif decision.promote:
                status = "would-promote (promote disabled)"
            else:
                status = "rejected"
        return 0
    except Exception as exc:
        log(f"FAILED: {exc}")
        decision = Decision(False, [str(exc)])
        return 1
    finally:
        minutes = (time.time() - t0) / 60
        if cand is not None and prod is not None:
            try:
                run_dir = _record(cfg, REPO / "finetune" / "loop" / day, cand, prod,
                                  decision, n_new, mine_rep, status, minutes)
                log(f"recorded {run_dir}/report.md")
                subprocess.run(["git", "add", "finetune/distilled", "finetune/picks", "finetune/loop"],
                               cwd=REPO, capture_output=True)
                subprocess.run(["git", "commit", "-m", f"data: nightly golden set {day} ({status})"],
                               cwd=REPO, capture_output=True)
            except Exception as exc:
                log(f"record/commit failed (non-fatal): {exc}")
        else:
            log(f"run ended: status={status} after {minutes:.0f} min (no eval pair to record)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distill-cap", type=int, default=40)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--time-budget-min", type=int, default=110)
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny train + 30-turn evals")
    args = ap.parse_args()
    cfg = LoopConfig(
        distill_cap=args.distill_cap, distill_model=args.model,
        time_budget_min=args.time_budget_min, promote=not args.no_promote,
        smoke=args.smoke,
    )
    raise SystemExit(run_nightly(cfg))


if __name__ == "__main__":
    main()
