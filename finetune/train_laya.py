"""CLI: fine-tune a laya checkpoint on the golden-path dataset.

Usage:
    uv run python finetune/train_laya.py [--base convaiinnovations/laya]
        [--data-dir finetune/data] [--out finetune/checkpoints/v1]
        [--max-len-long 2048] [--epochs-short 3] [--epochs-long 2] [--smoke]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_paths import data_file

from dataset import load_examples
from laya_backend import DEFAULT_BASE, load_checkpoint
from trainer import TrainConfig, run_training

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--data-dir", type=Path, default=data_file("finetune", "data"))
    ap.add_argument("--out", type=Path, default=data_file("finetune", "checkpoints", "v1"))
    ap.add_argument("--max-len-short", type=int, default=512)
    ap.add_argument("--max-len-long", type=int, default=2048)
    ap.add_argument("--epochs-short", type=int, default=3)
    ap.add_argument("--epochs-long", type=int, default=2)
    ap.add_argument("--batch-short", type=int, default=16)
    ap.add_argument("--batch-long", type=int, default=4)
    ap.add_argument("--max-long-rows", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    train = load_examples(args.data_dir / "train.jsonl")
    dev = load_examples(args.data_dir / "dev.jsonl")
    print(f"train={len(train)} dev={len(dev)} rows; base={args.base}")

    bundle = load_checkpoint(args.base)
    cfg = TrainConfig(
        out_dir=args.out,
        max_len_short=args.max_len_short,
        max_len_long=args.max_len_long,
        epochs_short=args.epochs_short,
        epochs_long=args.epochs_long,
        batch_short=args.batch_short,
        batch_long=args.batch_long,
        max_long_rows=args.max_long_rows,
        smoke=args.smoke,
    )
    result = run_training(bundle, _catalog(), train, dev, cfg)
    print(f"\ndone -> {result.out_dir}")
    for k, v in result.stage_metrics.items():
        print(f"  {k}: {v}")


def _catalog():
    import json
    return json.load(open(data_file("skills.json")))


if __name__ == "__main__":
    main()
