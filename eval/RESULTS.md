# Eval results

Date: 2026-09-23. Machine: local Mac, MPS. Checkpoint: `convaiinnovations/laya`
(English root, 421M) unless noted.

## Question: can laya, given a turn's opening prompt, predict which skills
that turn will load?

Dataset: 265 user turns mined from 68 pi session logs (7004 messages). 59
turns carry routable skill-load labels (loaded via the sanctioned `read`
tool); 14 dead-skill labels dropped (skills no longer on disk).

## Headline numbers (k=3, threshold 0.3, full 124-skill catalog)

| Metric | Value |
| --- | --- |
| label recall@3 | **0.174** |
| label recall@1 | 0.072 |
| label recall@5 | 0.208 |
| label recall@10 | 0.267 |
| turn recall@3 (all labels in top-3) | 0.074 (v1) |
| precision@3 | 0.073 |
| median gold rank | 21 of 124 |
| picks on no-load turns (k=10, thr 0.3) | 8.2 |
| sidecar latency p50 | 3105 ms (max 11.6s) |

## Why it fails, precisely

1. **Skill needs materialize mid-turn.** 53 of 59 labeled turns have their
   first skill read more than 6 tool calls into the turn window (9 windows
   exceed 50 tool calls). The load decision is driven by tool results and
   conversation evolution that no turn-start router can see. The status-quo
   design (skill list always in context, model reads on demand at any step)
   is structurally better suited to this pattern.
2. **Short prompts carry no signal.** Continuation turns ("ok I've clicked
   it") are unpredictable from text alone.
3. **Probability crowding.** Generic skills score p(core) 0.3-0.9 on nearly
   any prompt, so thresholds don't separate; at thr=0.3 the router fires on
   essentially every turn.
4. **Latency.** 124 questions x real prompts is ~3s p50 on MPS - too slow
   for turn-start even if accuracy were fine.

Dev-state sanity checks (5 crafted prompts) showed mean MRR ~0.50 across
all three laya checkpoints (english 0.495 / multilingual 0.505 /
typed-decisions 0.460) - real logs are much harder than crafted states.

## Verdict

Turn-start injection of laya's top-3 is a net negative for this workflow:
it would add ~3-8 irrelevant skill bodies per turn while catching ~17% of
what actually gets loaded. The main model's self-triggering (with all 124
name+description entries always in its prompt) remains the better
mechanism for these logs.

## What was shipped anyway

- The full harness (sidecar, extension, miner, eval) works end to end.
- The extension now defaults to **observe mode**: it scores every turn,
  logs decisions to `~/.pi/agent/laya-router/log.jsonl`, shows picks in the
  footer, but injects nothing. Set `LAYA_ROUTER_MODE=inject` to enable
  injection.
- Observe-mode logs accumulate the live prompt->picks record needed for any
  future rerun of this eval against real traffic.

## Paths that could still make this useful

- **Mid-turn routing**: score skills when the agent pauses between tool
  calls, using the accumulated transcript as state (the information that
  actually predicts loads). Bigger engineering, addresses cause #1.
- **Model routing instead of skill routing**: laya ships a router preset
  for small-vs-frontier model choice; pi's per-turn model switching could
  consume that directly.
- **Finetuning**: the answer space is defined at request time, but a
  fine-tuned checkpoint on (prompt -> loaded skill) pairs from observe-mode
  logs could lift recall well above 0.17. The dataset is already being
  collected.
