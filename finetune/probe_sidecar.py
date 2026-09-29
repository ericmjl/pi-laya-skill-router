"""Probe the production route path exactly as the pi extension does.

Starts nothing: expects the sidecar already running (LAYA_MODEL set to the
checkpoint under test). Sends real prompts with the real catalog via
POST /route — the same endpoint the extension hits on every turn — and
prints the picks.

Usage: uv run python finetune/probe_sidecar.py "prompt 1" "prompt 2" ...
"""

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data_paths import data_file

SIDECAR = "http://127.0.0.1:7699/route"

PROBES = [
    # (prompt, expected skill) — the expectation comes from the skill's own
    # description; a human judges the output, this list just standardizes it.
    "help me commit my changes across a bunch of files as logical commits",
    "put this on my calendar using the gog CLI, recurring Sundays with invites",
    "ingest this youtube video into my vault and make a summary note",
    "my stripe webhook returns 403 on live mode, help me debug the invoice",
    "what did I eat today and how many calories",
    "reply ok",
]


def route(prompt: str, skills: list[dict]) -> dict:
    payload = json.dumps(
        {
            "state": prompt,
            "skills": [{"name": s["name"], "description": s["description"]} for s in skills],
            "threshold": 0.3,
            "top_k": 3,
        }
    ).encode()
    req = urllib.request.Request(SIDECAR, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> None:
    prompts = sys.argv[1:] or PROBES
    skills = json.load(open(data_file("skills.json")))
    for p in prompts:
        r = route(p, skills)
        picks = ", ".join(
            f"{x['name']}({float(x['p']):.2f})" for x in r["picks"]
        )
        print(f'"{p[:60]}" -> [{picks}]  ({r["latency_ms"]}ms)')


if __name__ == "__main__":
    main()
