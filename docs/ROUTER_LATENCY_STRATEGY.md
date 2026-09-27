# Router latency strategy: right skill, right time, shortest latency

Problem: [#1](https://github.com/ericmjl/pi-laya-skill-router/issues/1) — the
router adds ~1–3.5s to every turn on `before_agent_start`, and 36% of routes
time out with no injection at all.

This doc is the plan. Every claim below is measured on this machine (M4 Max,
MPS) or computed from the production log — nothing is guessed. Options the
measurements killed are listed as dead ends so we don't revisit them.

## Measurements that decide the design

**Production** (`~/.pi/agent/laya-router/log.jsonl`, 217 routes / 123 errors):
p50 2164ms, p95 3245ms, 123 TimeoutErrors (36%). All errors are the
extension's fetch timeout firing — the slowest routes are also the ones that
return nothing.

**Row scaling, in-process, warm, 700-char state, real predict path:**

| rows | 1 | 8 | 12 | 16 | 24 | 40 | 110 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 85 | 151 | 233 | 316 | 457 | 811 | 2716 |

Cost is linear at ~19.5ms/row plus ~30ms fixed. The catalog is ~110–118
skills, so "score everything every turn" is a ~2.7s bill by construction.

**Dead end 1 — tokenization dedup.** The issue suspected the per-row state
re-encoding. Measured: `build_sequence` costs 0.24ms/row → 26ms of the 2700ms
pass (1%). Deduplicating it buys nothing. Row count is the only compute lever
that matters.

**Dead end 2 — fp16 on MPS.** The SDK forces fp32 on MPS (autocast is
CUDA-only, see `_amp_context`), and `model.half()` crashes MPS outright
(Metal dtype-mixing assertion in matmul). Constant-factor wins on MPS are
blocked without an ONNX/CoreML export. Revisit only if the remaining latency
still matters after the plan below.

**Dead end 3 — lexical shortlist.** BM25 over name+description recalls only
0.595 of gold labels at shortlist-24 — **below** what full-catalog laya
already delivers at top-3 (0.623 recall@3, fine-tuned v1). A lexical prefilter
would silently destroy quality. The misses are exactly the hard cases:
short continuations ("ok I've clicked it.") and inference-heavy matches.

**Dead end 4 — zero-shot dense embeddings.** Using laya's own ModernBERT as
a free embedder (mean-pool): recall@24 = 0.333. The cross-encoder's
representation space is not a retrieval space. A stage-1 ranker must be
trained (distilled), not reused.

**Dead end 5 — a static "working set".** Top-16 most-picked skills cover only
69% of production picks; 84 distinct skills have been picked at least once.
Popularity can only ever be a union term, never the shortlist itself.

**Safe facts the plan builds on:**

- `agent.predict` is deterministic (verified: identical ranked list and
  probabilities on repeated calls) → caching is exactly-safe.
- The extension already tracks `injectedThisSession`; skills already in
  context cost nothing if the router misses them. The real target is "NEW
  skills needed this turn" — a smaller target than full reranking.
- pi's `input` event fires the instant the user submits, before
  `before_agent_start`; `pi.sendMessage` while a run is streaming **steers**
  the active run (message lands before the model's next request). Latency can
  therefore be *hidden*, not only reduced.
- `collate_items` pads to batch max, not to `max_len` — trimming the 512/192
  token budgets buys nothing on typical states (state ≈ 193 tokens).

## The plan, in order of risk

### 1. Hide the latency (ships without touching model behavior)

Three changes, all in the extension + sidecar, none of which alter *which*
skills get picked:

- **Prefetch at `input`.** The route fetch fires the moment the user hits
  enter. `before_agent_start` awaits the in-flight promise instead of
  starting it.
- **Bounded block.** `before_agent_start` waits at most
  `LAYA_ROUTER_BLOCK_BUDGET_MS` (default 450ms). Within budget → inject
  exactly as today (first-request visibility). Past budget → return
  immediately (0ms added to the turn) and let the fetch finish in the
  background.
- **Steer, don't drop.** A background-resolved route calls
  `pi.sendMessage(...)`: mid-run it steers the active turn (model sees the
  skills at its next request — most real turns have tool loops, so little is
  lost); between turns it appends persistent context. Today those turns get
  *nothing* after a 3.5s stall. A stale route (a newer turn already started)
  is discarded via a generation counter.
- **Exact-match LRU cache** in the sidecar, keyed on
  (state, skill-list fingerprint, threshold, top_k, checkpoint). Hit rate is
  modest for novel prompts but retries and re-sends become free.
- **Warmup + keep-alive.** Startup warmup over the batch shapes the cascade
  will use (1/8/16/24/110 rows), then a background ping every ~25s. This
  kills the idle-stall tail (kernel recompile) that pushes production p50
  well above the warm number, and the 36% timeout rate with it.

Turn-start cost after this layer: cache hits ~0ms; everything else capped at
450ms with the route still landing mid-turn. No quality change — same
questions, same model, same picks.

### 2. Two-stage scoring (the structural fix, needs one trained model)

Stage-1: a **dual-tower bi-encoder distilled from the router itself**
(`finetune/train_biencoder.py`). The v1 cross-encoder scores (state, skill)
pairs; the bi-encoder learns to reproduce that ranking with one state encode
(~16ms) against ~110 cached skill-document embeddings (refreshed only when
the catalog changes). Stage-2: the existing laya pass on the shortlist —
**the same calibrated scorer, over ~16–24 rows instead of 110** (~316–457ms).

Why distillation is the requirement: stages 3–5 above prove the shortlist
cannot be lexical or zero-shot. Distilling the cross-encoder's own rankings
on the golden dataset (292 labeled turns, growing nightly) is the
standard fix, and the repo already owns every input: labels
(`dataset.turn_truth` is the single source of truth), the teacher
(`predict_ranking` through `checkpoints/v1`), and the harness.

Quality guardrails, in order:

1. **Promotion gate** — the cascade runs the same `run_eval.py` harness as
   v1; recall@1/3/5, precision@3, no-load picks, median gold rank must match
   full-catalog within noise (recall@3 within ±0.02) or it doesn't ship.
2. **Confidence fallback** — if the bi-encoder's top cosine is below
   `LAYA_S1_MIN_SIM`, the sidecar runs the full catalog for that turn. Worst
   case falls back to today's quality at today's cost, only on
   out-of-distribution states.
3. **Nightly monitoring** — the auto-learning loop already re-mines labels;
   it measures stage-1 shortlist recall on every new labeled turn and alerts
   on drift.
4. **Kill switch** — `LAYA_SHORTLIST_K=0` restores full-catalog behavior
   without touching code.

Expected steady state: ~16ms (stage-1) + ~320ms (stage-2 at 16 rows) ≈
**~340ms warm vs ~2700ms today — 8×**, with the 450ms block budget usually
met and mid-run steering as the safety net.

### Measured: the gate passes (2026-09-26)

Bi-encoder trained per the recipe above (`finetune/checkpoints/biencoder-v1`;
322 teacher-scored states, 14 epochs, held-out shortlist recall@16 0.820,
@24 0.918). Full `run_eval.py` harness through the real sidecar, same
265-turn mined dataset, same checkpoint (v1), same day, same catalog:

| configuration | recall@3 | precision@3 | no-load picks | gold rank med | sidecar p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full catalog (baseline) | 0.623 | 0.270 | 3.73 | 2 | 1532 ms |
| **cascade k=16** | **0.684** | **0.296** | **2.65** | 2 | **199 ms** |
| cascade k=24 | 0.679 | 0.289 | 3.01 | 2 | 264 ms |

The cascade is not a trade — it is better on every quality axis while
running ~7.7× faster. Removing ~90 distractor rows gives the cross-encoder
a cleaner candidate set: gold ranks higher, fewer false positives survive
the threshold. The warmup + keep-alive alone also pulled the full-catalog
p50 down from the 2.7s in-process measurement to 1.5s, worth noting for
any deployment that keeps `LAYA_SHORTLIST_K=0`.

Promotion status: **passed**. The remaining step before flipping production
is one observe-mode day with the cascade enabled to compare live footer
picks against today's log, then setting `LAYA_BIENCODER` +
`LAYA_SHORTLIST_K=16` in the launchd plist.
### Production config (live 2026-09-26)

k=24, threshold 0.4, `use_desc` on. The threshold sweep shows the operating
curve is flat from 0.25-0.45 (recall@3 constant, precision rising, no-load
picks falling), and descriptions in the question — `use_desc=true`, which
v1 was trained to be robust to — measured through the full harness:

| cascade k=24, thr 0.4 | recall@1 | recall@3 | recall@5 | precision@3 | no-load picks | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bare question | — | 0.679 | 0.778 | 0.289 | 3.01 | 264 ms |
| **with descriptions** | **0.448** | **0.736** | **0.840** | **0.302** | **1.91** | 961 ms |

Descriptions cost ~3.6x stage-2 latency (longer question rows), which is
why the extension runs observe-mode routing at zero block budget: the
route result only feeds the footer and log, so it lands via the late path
and costs the turn nothing. In inject mode, re-evaluate the budget
trade-off before enabling `LAYA_ROUTER_USE_DESC`.

Live-probe note (why k=24 and not 16): the commit-skills probe ranked
`atomic-commits` at stage-1 position 17 — just outside k=16. k=24 is a
harness tie with headroom for exactly this tail. The stripe hard case is
shortlisted but still scored below threshold by stage-2: a data problem
(hard negatives / more labels), not a retrieval one.

### 3. The one-question gate (quality and latency at once)

v1's residual weakness is false positives on turns that need no skill
(3.7 picks/turn on no-load turns). That is also a latency problem: every
no-skill turn still pays the full pass. A **single-row laya question** —
"does this turn need any of the loaded skills?" — costs ~85ms, and the
golden dataset already contains 120 explicit none-needed turns to train it.
Gate first (1 row); if yes, run the shortlist pass. Turns needing nothing
resolve at <100ms, and the no-load false-positive load drops with it. Train
and gate it through the same promotion-gate protocol.

### 4. Optional later: ONNX/CoreML export

The SDK ships `onnx_agent.py`; CoreML EP would reclaim fp16 internally and
flatten the MPS tail. Only worth it if, after 1–3, the remaining ~300ms
still hurts. It is a constant factor; 1–3 are structural.

## What ships now (this branch)

- Sidecar: exact-cache, multi-shape warmup, keep-alive, env-gated shortlist
  endpoint plumbing, stage timings in the response and log. With no
  bi-encoder configured, behavior is byte-identical to today except faster
  tails.
- Extension: `input` prefetch, 450ms bounded block, steer-on-late via
  `pi.sendMessage`, generation-counter staleness guard, richer route logs
  (`path`, `cached`, `s1_ms`, `s2_ms`).
- `finetune/train_biencoder.py`: teacher-score precompute + dual-tower
  training + shortlist-recall eval, wired to `dataset.turn_truth` and
  `laya_backend.predict_ranking`. Promotion stays manual until the gate in
  `eval/` passes.

## Eval protocol (unchanged, now load-bearing)

```bash
uv run scripts/scan_skills.py
uv run scripts/run_eval.py --k 3 --threshold 0.3          # full catalog (baseline)
LAYA_SHORTLIST_K=16 LAYA_BIENCODER=finetune/checkpoints/biencoder-v1 \
  uv run python sidecar/server.py &                        # cascade
uv run scripts/run_eval.py --k 3 --threshold 0.3          # must match within noise
```

The harness routes every mined turn through the real HTTP sidecar, so the
cascade is evaluated exactly as production would run it.
