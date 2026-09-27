"""CLI: run golden-path distillation over all pi sessions.

Usage:
    uv run python finetune/distill.py [--model glm-5.3-flash] [--workers 4]
        [--limit N] [--force] [--sessions-dir PATH]

Resumable: per-session caches land in finetune/distilled/ and are reused
unless --force. Safe to re-run after failures.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from distiller import DEFAULT_MODEL, distill_turns
from session_data import DEFAULT_SESSIONS_DIR, load_turns

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-chars", type=int, default=300_000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=None, help="only first N sessions (smoke)")
    ap.add_argument("--force", action="store_true", help="ignore caches")
    ap.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    args = ap.parse_args()

    skills = json.load(open(REPO / "skills.json"))
    turns = load_turns(args.sessions_dir)
    if args.limit:
        keep = sorted({t.session for t in turns})[: args.limit]
        turns = [t for t in turns if t.session in keep]

    print(f"distilling {len(turns)} turns with {args.model} (workers={args.workers})")
    golden, report = distill_turns(
        turns, skills, out_dir=REPO / "finetune" / "distilled",
        model=args.model, workers=args.workers, budget_chars=args.budget_chars,
        timeout_s=args.timeout, force=args.force,
    )
    print(report.summary())
    if report.failures:
        print("\nfailed sessions:")
        for f in report.failures[:10]:
            print(f"  {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
