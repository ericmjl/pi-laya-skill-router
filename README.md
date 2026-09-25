# pi-laya-skill-router

A pi coding-agent extension that uses the [laya](https://huggingface.co/convaiinnovations/laya)
System-1 decision model (421M params, ~35ms per forward pass, calibrated
probabilities, runs locally on Apple MPS) to decide which agent skills are
relevant to each request, and pre-loads them into the model's context.

Normally pi lists every skill's name and description in the system prompt and
trusts the main model to `read` the SKILL.md when relevant. That fails
silently on weaker models and costs a tool round-trip when it works. This
router scores all skills on every turn and injects the winners verbatim.

## Components

| Path | What it is |
| --- | --- |
| `sidecar/server.py` | FastAPI service wrapping laya. Loads the checkpoint once, scores skill batches in one forward pass. |
| `extension/index.ts` | pi extension. On every `before_agent_start`, routes the prompt through the sidecar and injects the top-k skills as a visible message. |
| `scripts/scan_skills.py` | Collects name+description from all SKILL.md frontmatter into `skills.json`. |
| `scripts/mine_sessions.py` | Builds the eval dataset from pi session logs (ground truth: which skills the model actually loaded via the `read` tool). |
| `scripts/run_eval.py` | Scores the dataset through the sidecar; reports recall/precision@k and false-positive rate on no-load turns. |
| `scripts/template_experiment.py` | Question-template A/B harness against labeled dev states. |
| `scripts/checkpoint_compare.py` | Compares laya checkpoints (english / multilingual / typed-decisions) on the dev states. |
| `finetune/` | Golden-path distillation + SFT pipeline: frontier-model labeling of session traces, dataset build, MPS training, offline eval. See `finetune/README.md` and `eval/FINETUNE_RESULTS.md`. |

## Install

```bash
# 1. the extension (registers the skill router with pi)
pi install git:github.com/ericmjl/pi-laya-skill-router

# 2. local services: sidecar + skill scan (add --with-nightly for the
#    auto-learning loop)
git clone https://github.com/ericmjl/pi-laya-skill-router ~/pi-laya-skill-router
cd ~/pi-laya-skill-router && ./install.sh

# 3. restart pi
```

Routing runs entirely on your machine. The optional nightly fine-tune loop
sends transcript excerpts one-way to your configured Anthropic access for
golden-path labeling; session data itself never leaves your machine and all
learned artifacts are gitignored by default.

Optional launchd auto-start: `./install.sh` handles this (the repo's plists
are `__HOME__`/`__UV__` templates; install.sh substitutes and loads them).

## Config (env vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `LAYA_ROUTER_MODE` | `observe` | `observe` (log only) or `inject` (inject top-k bodies) |
| `LAYA_ROUTER_URL` | `http://127.0.0.1:8787/route` | Sidecar endpoint |
| `LAYA_ROUTER_THRESHOLD` | `0.3` | Min p(core) to pick a skill |
| `LAYA_ROUTER_TOP_K` | `3` | Max skills injected per turn |
| `LAYA_ROUTER_MAX_SKILL_CHARS` | `8000` | Per-skill body cap |
| `LAYA_ROUTER_TIMEOUT_MS` | `2500` | Route call timeout (fail-open) |
| `LAYA_MODEL` | `convaiinnovations/laya` | Checkpoint |
| `LAYA_PORT` | `8787` | Sidecar port |

Note: ports 8771/8772 are occupied by other local services on this machine.

## Behavior details

- **Default mode is `observe`**: every turn is scored and logged, picks show
  in the footer, but nothing is injected. Set `LAYA_ROUTER_MODE=inject` to
  enable injection. The eval (eval/RESULTS.md) found turn-start injection a
  net negative for this workflow; observe mode collects the live data any
  redesign or fine-tune would need.
- Decisions run every turn; injected bodies are not re-sent within a session
  (persistent messages stay in the transcript). After `/compact` the set
  clears so skills can be re-injected if needed.
- Explicit `/skill:name` invocations bypass the router entirely.
- Sidecar down = fail-open: turns proceed without injection; footer shows
  `laya-router: sidecar down`.
- Every decision is logged to `~/.pi/agent/laya-router/log.jsonl`.
- `/routerstats` prints aggregate stats from the log.

## Eval

```bash
uv run scripts/scan_skills.py
uv run scripts/mine_sessions.py
uv run python sidecar/server.py &   # or launchd
uv run scripts/run_eval.py --k 3 --threshold 0.3
```

Dataset: every user turn in `~/.pi/agent/sessions` (265 turns across 68
sessions), labeled with the skills loaded via the sanctioned `read` tool in
that turn. See `eval/RESULTS.md` for current numbers.

## Known caveats

- laya's zero-shot accuracy on novel task families is modest; dev-state mean
  MRR is ~0.50 across all three checkpoints. The router is a top-k ranker,
  not a magic relevance oracle: it reliably surfaces lexically-obvious
  matches and narrows 111 skills to 3, but inference-heavy matches
  (e.g. "stripe webhook fails" -> `stripe-cli-billing-ops`) can be missed.
- First predict after sidecar start includes MPS kernel warmup (~500ms);
  steady state is ~1s for the full 111-skill catalog.
- laya ships a checkpoint warning about invalid calibration temperatures for
  questions with 11+ options; binary/ternary criteria questions are
  unaffected.

## Fine-tuning

The zero-shot router above was fine-tuned on golden-path labels distilled
from pi session logs (frontier-model `should_have` judgments + observed
reads). Headline: recall@3 0.174 → **0.623** through the same harness
protocol, no-load picks 8.2 → 3.7, median gold rank 21 → 2, and a 2048-token
context stage that keeps selection working on tool-rich states where the
shipped 512-token checkpoint collapses to zero. Full evidence:
`eval/FINETUNE_RESULTS.md`. Rerun/extend the loop with `finetune/README.md`.
