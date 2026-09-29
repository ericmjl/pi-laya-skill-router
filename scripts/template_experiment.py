"""Question-template experiment for laya skill routing.

Scores a handful of labeled dev states against the full skill catalog
under several question templates, and reports MRR + impostor gaps.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_paths import data_file

os.environ.setdefault("USE_TF", "0")

import laya

DEV_STATES = [
    (
        "help me ingest the meeting notes into my vault then draft the retreat vendor follow-up email",
        ["vault-meeting-notes-ingestion", "gog-gmail-cli"],
    ),
    ("what did I eat today", ["food-journal-daily-log"]),
    (
        "publish my blog post about the quantum amplitudes widget to the website",
        ["website-blog-publishing", "blogbot", "write-like-eric"],
    ),
    ("the stripe webhook keeps failing with a signature error in CI", ["stripe-cli-billing-ops"]),
    ("quiz me on what we discussed about attention heads", ["socratic-quizzing-pedagogy"]),
]

TEMPLATES = {
    "T1_help": lambda name, desc: {
        "type": "choice",
        "instructions": (
            f"Decide whether the skill '{name}' should be loaded to help answer the user's request."
        ),
        "criteria": {
            "relevant": f"the skill clearly helps: {desc}" if desc else "the skill clearly helps with the request",
            "not relevant": "the skill would not help with this request",
        },
    },
    "T2_core": lambda name, desc: {
        "type": "choice",
        "instructions": (
            f"Is loading the skill '{name}' necessary to handle this request properly?"
        ),
        "criteria": {
            "necessary": "this request is squarely this skill's job; skipping it would mishandle the request",
            "not necessary": "at best tangentially related; the request can be handled well without it",
        },
    },
    "T3_three": lambda name, desc: {
        "type": "choice",
        "instructions": f"How related is the skill '{name}' to this request?",
        "criteria": {
            "core": "the request is exactly what this skill exists for",
            "tangential": "somewhat related but the request does not really need it",
            "unrelated": "no meaningful connection to the request",
        },
    },
}


def mrr(ranked_names: list[str], relevant: list[str]) -> float:
    best = min((ranked_names.index(r) + 1) for r in relevant if r in ranked_names) if any(
        r in ranked_names for r in relevant
    ) else None
    return 1.0 / best if best else 0.0


def main() -> None:
    skills = json.load(open(data_file("skills.json")))
    catalog = {s["name"]: s["description"][:220] for s in skills}
    agent = laya.load("convaiinnovations/laya", device="mps")

    for tname, make_q in TEMPLATES.items():
        print(f"\n===== {tname} =====")
        mrrs = []
        for state, relevant in DEV_STATES:
            questions = {n: make_q(n, d) for n, d in catalog.items()}
            result = agent.predict(state, questions)
            scored = sorted(
                ((n, result["answers"][n]["probabilities"].get("core" if tname == "T3_three" else "necessary" if tname == "T2_core" else "relevant", 0.0)) for n in catalog),
                key=lambda x: x[1],
                reverse=True,
            )
            names = [n for n, _ in scored]
            m = mrr(names, relevant)
            mrrs.append(m)
            top5 = ", ".join(f"{n}({p:.2f})" for n, p in scored[:5])
            gold_ps = {n: dict(scored)[n] for n in relevant if n in catalog}
            print(f"  MRR={m:.2f}  gold={{{', '.join(f'{n}:{p:.2f}' for n, p in gold_ps.items())}}}")
            print(f"    top5: {top5}")
        print(f"  --> mean MRR: {sum(mrrs)/len(mrrs):.3f}")


if __name__ == "__main__":
    main()
