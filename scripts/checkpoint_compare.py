"""Compare laya checkpoints on the skill-routing dev states (T3 template)."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_paths import data_file

os.environ.setdefault("USE_TF", "0")

import laya

from template_experiment import DEV_STATES

TEMPLATE = lambda name, desc: {
    "type": "choice",
    "instructions": f"How related is the skill '{name}' to this request?",
    "criteria": {
        "core": "the request is exactly what this skill exists for",
        "tangential": "somewhat related but the request does not really need it",
        "unrelated": "no meaningful connection to the request",
    },
}


def mrr(ranked, relevant):
    ranks = [ranked.index(r) + 1 for r in relevant if r in ranked]
    return 1.0 / min(ranks) if ranks else 0.0


def main():
    repo = os.path.expanduser("~/github/pi-laya-skill-router")
    skills = json.load(open(data_file("skills.json")))
    catalog = {s["name"]: s["description"][:220] for s in skills}

    for model_id in sys.argv[1:]:
        print(f"\n########## {model_id} ##########", flush=True)
        try:
            agent = laya.load(model_id, device="mps")
        except Exception as e:  # noqa: BLE001
            print(f"  load failed: {e}")
            continue
        mrrs = []
        for state, relevant in DEV_STATES:
            questions = {n: TEMPLATE(n, d) for n, d in catalog.items()}
            result = agent.predict(state, questions)
            scored = sorted(
                ((n, result["answers"][n]["probabilities"].get("core", 0.0)) for n in catalog),
                key=lambda x: x[1],
                reverse=True,
            )
            names = [n for n, _ in scored]
            m = mrr(names, relevant)
            mrrs.append(m)
            print(f"  MRR={m:.2f} top5: " + ", ".join(f"{n}({p:.2f})" for n, p in scored[:5]), flush=True)
        print(f"  --> mean MRR: {sum(mrrs)/len(mrrs):.3f}", flush=True)


if __name__ == "__main__":
    main()
