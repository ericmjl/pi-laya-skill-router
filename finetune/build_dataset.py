"""CLI: merge observed + golden labels into training examples.

Usage:
    uv run python finetune/build_dataset.py [--distilled-dir PATH] [--out-dir PATH]

Writes train.jsonl / dev.jsonl / test.jsonl and prints stats. Runs fine
with no distilled labels yet (observed reads only) — the smoke path.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import build_dataset, load_golden, write_jsonl
from mine_picks import load_picks
from session_data import load_turns

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distilled-dir", type=Path, default=REPO / "finetune" / "distilled")
    ap.add_argument("--picks-dir", type=Path, default=REPO / "finetune" / "picks",
                    help="mine_picks cache dir; the router's own joined picks")
    ap.add_argument("--out-dir", type=Path, default=REPO / "finetune" / "data")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    skills = json.load(open(REPO / "skills.json"))
    turns = load_turns()
    golden = load_golden(args.distilled_dir)
    picks = load_picks(args.picks_dir)
    n_golden_turns = sum(len(v) for v in golden.values())
    print(f"turns: {len(turns)}  golden turns: {n_golden_turns} across {len(golden)} sessions  "
          f"picked turns: {len(picks)}")

    examples, stats = build_dataset(turns, golden, skills, picks=picks, seed=args.seed)
    for split in ("train", "dev", "test"):
        subset = [e for e in examples if e.split == split]
        write_jsonl(subset, args.out_dir / f"{split}.jsonl")
        by = Counter((e.state_kind, e.target) for e in subset)
        print(f"{split}: {len(subset)} rows | " +
              " ".join(f"{k}={v}" for k, v in sorted(by.items())))
    print("stats:", json.dumps(stats))


if __name__ == "__main__":
    main()
