# Fine-tuning the laya skill router

Automatic procedure for turning pi session traces + frontier-model judgment
into a fine-tuned laya checkpoint, covering:

- **(a) context window** — stage-2 training at longer `max_len` (ModernBERT
  natively supports 8192; the shipped checkpoint serves 512)
- **(b) skill selection matched to a frontier model** — golden-path
  distillation, then SFT on the distilled labels
- **(c) evidence in the pi harness** — held-out eval through the same
  protocol as `eval/RESULTS.md`, plus live observe-mode probes

## Architecture

Deep modules, one policy each, dataclass contracts across boundaries:

| Module | Interface | Owns |
| --- | --- | --- |
| `router_q.py` | `build_question`, criteria constants | the train/serve format contract — the only place criteria wording lives |
| `session_data.py` | `load_turns`, `render_windows` | pi session-JSONL parsing, turn windowing, SKILL-read annotation |
| `distiller.py` | `distill_turns` | frontier-model labeling: prompt policy, pi CLI (headless, tool-less), validation, caching |
| `dataset.py` | `build_dataset`, `turn_truth`, `load_golden` | the ONE merge of observed+golden labels, sampling, session-level splits |
| `laya_backend.py` | `load_checkpoint`, `save_checkpoint`, `encode_examples`, `predict_ranking` | checkpoint format, tokenization, stock-SDK-faithful prediction |
| `trainer.py` | `run_training` | curriculum stages, optimizer groups, CE objective, dev metrics, export |
| `train_biencoder.py` | teacher scores, `BiEncoder`, `train`, `eval_split` | stage-1 shortlist distillation: dual-tower student of the cross-encoder, full-catalog InfoNCE + distribution match, shortlist-recall eval |
| `eval_ckpt.py` | `evaluate`, `reports_to_md` | offline metrics vs observed AND golden references, ablations |

## The golden-path procedure (why distillation)

Ground truth mined from `read` calls (the old `mine_sessions.py` labels) is
doubly incomplete: skills materialize mid-turn driven by tool results (53/59
labeled turns read their first skill >6 tool calls in), and skills the main
model *should* have loaded but didn't never appear at all. The distiller
sends each session's compacted transcript — with `[SKILL-READ]` steps
pre-annotated — to a frontier model with the full skill catalog and gets
back per turn:

- `loaded` — confirmed reads (anchors the labeler to observed reality)
- `should_have` — skills the work needed, **with an evidence citation**
  (uncited entries are dropped at validation)
- `tangential` — near-misses, kept as a separate training class
- `when` — `turn_start` or `after_step_N`: the earliest point the need was
  inferable. Mid-turn `when` values are why long-context states matter.
- `none_needed` turns become the hard negatives that fix the probability
  crowding which sank the zero-shot router (8.2 picks on no-load turns).

## The loop

```bash
# 0. catalog (gitignored; regenerated)
uv run python scripts/scan_skills.py

# 1. distill golden paths (resumable; caches in finetune/distilled/)
uv run python finetune/distill.py --workers 4

# 2. build training rows (train/dev/test split by session hash)
uv run python finetune/build_dataset.py

# 3. SFT: stage 1 short@512, stage 2 long@2048; exports a laya-loadable dir
uv run python finetune/train_laya.py --out finetune/checkpoints/v1

# 4. offline evidence: baseline vs fine-tuned on held-out sessions,
#    with the context-length ablation on the fine-tuned checkpoint
uv run python finetune/eval_ckpt.py \
  --ckpt baseline=convaiinnovations/laya \
  --ckpt v1=finetune/checkpoints/v1 \
  --max-len 512,1024,2048 --split test

# 5. harness evidence: serve the fine-tuned ckpt through the real sidecar
LAYA_MODEL=$PWD/finetune/checkpoints/v1 uv run python sidecar/server.py &
uv run scripts/run_eval.py --k 3 --threshold 0.3   # same protocol as RESULTS.md
```

Retraining after new traffic: repeat 1-4. Distillation is cached per session,
so only new sessions cost frontier-model calls; `--force` relabels.

## Design decisions worth knowing

- **Question template is a contract** (`router_q.py`): train and serve must
  use byte-identical criteria wording or the checkpoint silently scores
  against a different rubric than it learned.
- **Every example is emitted in both question variants** (bare / with skill
  description) and both state kinds (short prompt / long tool-aware state),
  so one checkpoint serves either configuration.
- **Unknown ground truth never becomes fake negatives**: turns with no
  observed reads and no golden coverage are skipped, not counted as
  none-needed.
- **Split by session hash**, never by turn, or the same session leaks across
  train/test.
- **`predict_ranking` delegates to the stock Agent decode** so eval numbers
  are computed under the same temperature rules as production serving.
- **Export keeps the long `max_len` in the served config**; the ablation in
  step 4 quantifies the latency cost before that ships. Drop back to
  `--max-len-long 512` if the latency number is unacceptable.
- **Act head is frozen**: the router only consumes `p(core)`; freezing the
  action head preserves its calibration signal for other consumers.

## Rollout

The launchd sidecar (`launchd/com.ericmjl.laya-sidecar.plist`) serves the
production router. It expects the checkpoint at a durable machine path —
worktrees are ephemeral — so promote a trained checkpoint before flipping:

```bash
rsync -a --delete finetune/checkpoints/v1/ ~/.pi/agent/laya-router/checkpoints/v1/
cp launchd/com.ericmjl.laya-sidecar.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.ericmjl.laya-sidecar.plist
launchctl load ~/Library/LaunchAgents/com.ericmjl.laya-sidecar.plist
curl -s localhost:7699/health   # model should name the checkpoint path
```

The plist sets `LAYA_MODEL=~/.pi/agent/laya-router/checkpoints/v1` (installed with real paths by `install.sh`);
the extension keeps routing every turn and logging to
`~/.pi/agent/laya-router/log.jsonl`, which doubles as the hard-negative mine
for the next training round.

## The auto-learning loop

`launchd/com.ericmjl.laya-nightly.plist` runs `finetune/nightly_loop.py` at
3 AM. Each run distills only NEW sessions (per-session caches make this
incremental; `--distill-cap` bounds nightly API cost), retrains **from base**
on the full ever-expanding golden set (never from last night's checkpoint —
errors don't compound), evaluates candidate vs production on the deterministic
session-hash test holdout, and promotes only through a gate: candidate must
beat production on golden-recall@3, must not regress no-load picks, and must
stay inside the latency budget. Runs are recorded in `finetune/loop/`
(`history.jsonl` + per-run `report.md`) and the expanded golden set is
committed with each run. Failure anywhere never touches the live sidecar.

```bash
# manual run / dry-run
uv run python finetune/nightly_loop.py --no-promote
cp launchd/com.ericmjl.laya-nightly.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ericmjl.laya-nightly.plist
```

## Known limitations (v1)

- Negative sampling is uniform over the catalog; hard-negative mining from
  observe-mode false positives (`~/.pi/agent/laya-router/log.jsonl`) is the
  obvious v2 upgrade.- `when: after_step_N` labels are validated but the mid-turn *inference*
  wiring (routing between tool calls) is the bigger engineering item named
  in `eval/RESULTS.md`; this checkpoint trains the capability, the harness
  work to consume it is separate.
- Temperature calibration is inherited from the base checkpoint; refit on
  dev after SFT if calibration drift matters (ECE is in `laya.common`).
