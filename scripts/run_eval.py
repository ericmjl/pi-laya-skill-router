"""Score the mined eval dataset through the laya sidecar. V2.

V2 changes:
  - state includes the previous user turn from the same session (what the
    production router can legitimately see in-conversation)
  - labels split into "early" (first read within the turn's first 6 tool
    calls: prompt-predictable) vs "all"
  - recall@1/3/5/10, median gold rank, FP rate on no-load turns

Usage: uv run scripts/run_eval.py [--threshold 0.3] [--k 3]
Requires the sidecar running on :8787.
"""

import argparse
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIDECAR = "http://127.0.0.1:8787/route"
EARLY_TOOL_CALLS = 6


def route(state: str, skills: list[dict], threshold: float, k: int) -> dict:
    payload = json.dumps(
        {
            "state": state[:1400],
            "skills": [{"name": s["name"], "description": s["description"]} for s in skills],
            "threshold": threshold,
            "top_k": k,
        }
    ).encode()
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(SIDECAR, data=payload, headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=300))
        except Exception as exc:  # noqa: BLE001 - sidecar may be restarting
            last_err = exc
            print(f"  route attempt {attempt + 1} failed ({exc}); backing off", flush=True)
            time.sleep(5 * (attempt + 1))
            _wait_for_health()
    raise RuntimeError(f"route failed after retries: {last_err}")


def _wait_for_health(timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(SIDECAR.replace("/route", "/health"), timeout=5) as r:
                if json.load(r).get("loaded"):
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dump", default="eval/scores_v2.jsonl")
    args = ap.parse_args()

    skills = json.load(open(REPO / "skills.json"))
    catalog = {s["name"] for s in skills}
    turns = [json.loads(l) for l in open(REPO / "eval" / "dataset.jsonl")]
    # labels pointing at skills that no longer exist on disk can never be
    # routed; count those turns as no-load instead
    dropped = 0
    for t in turns:
        routable = {s for s in t["labels"] if s in catalog}
        dropped += len(t["labels"]) - len(routable)
        t["labels"] = sorted(routable)
    print(f"eval turns: {len(turns)}, catalog: {len(catalog)} skills, thr={args.threshold}")
    print(f"dropped dead-skill labels: {dropped}")

    # group by session to build prev-turn context
    by_session: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        by_session[t["session"]].append(t)
    for tl in by_session.values():
        tl.sort(key=lambda t: t["ts"])
        for i, t in enumerate(tl):
            t["prev_state"] = tl[i - 1]["state"] if i > 0 else ""
            t["turn_index"] = i

    latencies = []
    out = open(REPO / args.dump, "w")
    ks = [1, 3, 5, 10]
    metrics: dict[str, list[float]] = defaultdict(list)
    t_start = time.time()
    for idx, t in enumerate(turns):
        positives = set(t["labels"]) or set(t["bash_labels"])
        state = t["state"] if len(t["state"]) > 60 else (t["prev_state"] + " || " + t["state"]).strip()
        r = route(state, skills, args.threshold, 10)
        latencies.append(r["latency_ms"])
        ranked = [p["name"] for p in r["all"]]
        picks = [p["name"] for p in r["picks"]]

        if positives:
            for k in ks:
                hits = positives & set(ranked[:k])
                metrics[f"label_recall@{k}"].append(len(hits) / len(positives))
            gold_ranks = [ranked.index(g) + 1 for g in positives if g in ranked]
            if gold_ranks:
                metrics["gold_rank"].extend(gold_ranks)
            # early slice: small turns are prompt-predictable
            if t["n_tool_calls"] <= EARLY_TOOL_CALLS:
                for k in ks:
                    hits = positives & set(ranked[:k])
                    metrics[f"early_label_recall@{k}"].append(len(hits) / len(positives))
                metrics["early_n"].append(1.0)
            else:
                metrics["late_n"].append(1.0)
            metrics["precision@3"].append(len(set(picks[:3]) & positives) / 3)
        else:
            metrics["noload_picks"].append(len(picks))

        out.write(
            json.dumps(
                {
                    "state": state[:120],
                    "labels": sorted(positives),
                    "turn_index": t["turn_index"],
                    "n_tool_calls": t["n_tool_calls"],
                    "gold_ranks": {g: ranked.index(g) + 1 for g in positives if g in ranked},
                    "top10": [(p["name"], round(p["p"], 3)) for p in r["all"][:10]],
                }
            )
            + "\n"
        )
        if (idx + 1) % 50 == 0:
            print(f"  ...{idx + 1}/{len(turns)} ({time.time() - t_start:.0f}s)", flush=True)
    out.close()

    print(f"\n=== results (thr={args.threshold}) ===")
    for m in sorted(metrics):
        vals = metrics[m]
        if m == "gold_rank":
            vals_sorted = sorted(vals)
            print(f"{m}: median={vals_sorted[len(vals_sorted)//2]} n={len(vals)}")
        elif m in ("early_n", "late_n"):
            print(f"{m}: {len(vals)}")
        else:
            print(f"{m}: mean={sum(vals)/len(vals):.3f} (n={len(vals)})")
    print(f"sidecar latency: p50={sorted(latencies)[len(latencies)//2]}ms max={max(latencies)}ms")


if __name__ == "__main__":
    main()
