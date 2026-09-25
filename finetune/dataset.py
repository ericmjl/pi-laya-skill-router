"""Golden labels + observed reads -> training examples for the router SFT.

Deep-module contract:
    build_dataset(turns, golden, skills, ...) -> list[Example]
    write_jsonl / load_examples

Merge policy (the interesting part — everything else is plumbing):
  positives  observed reads ∪ golden.loaded ∪ golden.should_have  -> target core
  near-miss  golden.tangential                                    -> target tangential
  negatives  catalog minus the above, sampled per turn            -> target unrelated
  low-confidence golden turns and turns with no signal at all are skipped:
  unknown ground truth must not become fake negatives.

Each example is emitted in both question variants (bare / with description)
so the checkpoint is robust to the sidecar's `use_desc` flag, and in both
state kinds (short = user prompt; long = prompt + tool activity), which is
what the context-window curriculum trains on.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from router_q import CORE, TANGENTIAL, UNRELATED
from session_data import Turn, session_split

# Sampling sizes per turn kind.
NEG_PER_POSITIVE_TURN = 6     # turns that need skills: fixed negative floor
NEG_FLOOR_NONE_NEEDED = 8     # none-needed turns: suppression examples
MIN_SHORT_STATE = 60          # below this, prepend previous turn (run_eval v2 rule)


@dataclass
class Example:
    state: str
    state_kind: str      # short | long
    variant: str         # bare | desc
    skill: str
    target: str          # core | tangential | unrelated
    source: str          # observed | distilled | sampled
    split: str
    session: str
    when: str = "turn_start"
    evidence: str = ""


@dataclass
class Truth:
    """The single merge of observed + golden labels for one turn.

    Both the dataset builder and the offline eval go through this function —
    there is exactly one definition of ground truth in the codebase.
    `known` is False when we have no trustworthy signal (no observed reads
    and no usable golden entry); unknown turns are skipped by both consumers
    rather than becoming fake negatives.
    """

    positives: dict          # skill -> source (observed | distilled)
    tangentials: dict        # skill -> evidence
    known: bool

    @property
    def none_needed(self) -> bool:
        return self.known and not self.positives and not self.tangentials


def turn_truth(turn: Turn, g, catalog: set[str]) -> Truth:
    """Merge observed reads with one turn's GoldenTurn (g may be None)."""
    positives: dict[str, str] = {s: "observed" for s in turn.observed_labels if s in catalog}
    tangentials: dict[str, str] = {}
    known = bool(positives)
    if g is not None and g.confidence != "low":
        known = True
        for s in g.loaded + g.should_have:
            if s in catalog and s not in positives:
                positives[s] = "distilled"
        for s in g.tangential:
            if s in catalog and s not in positives:
                tangentials[s] = g.evidence.get(s, "")
    return Truth(positives=positives, tangentials=tangentials, known=known)


def _short_state(turn: Turn) -> str:
    state = turn.state
    if len(state.strip()) <= MIN_SHORT_STATE and turn.prev_state:
        return (turn.prev_state + " || " + state).strip()
    return state


def build_dataset(
    turns: list[Turn],
    golden: dict[str, dict[int, "object"]],
    skills: list[dict],
    seed: int = 13,
    splits=(0.7, 0.15, 0.15),
) -> tuple[list[Example], dict]:
    """golden: {session: {turn_index: GoldenTurn}} (missing sessions/turns =
    no signal). Returns (examples, stats)."""
    rng = random.Random(seed)
    catalog = {s["name"]: s for s in skills}
    by_session: dict[str, list[Turn]] = {}
    for t in turns:
        by_session.setdefault(t.session, []).append(t)

    examples: list[Example] = []
    stats = {"turns_pos": 0, "turns_none": 0, "turns_skipped": 0,
             "pos": 0, "tang": 0, "neg": 0}

    for session, sturns in by_session.items():
        split = session_split(session, splits)
        gsession = golden.get(session, {})
        for turn in sturns:
            truth = turn_truth(turn, gsession.get(turn.turn_index), set(catalog))
            if not truth.known:
                stats["turns_skipped"] += 1
                continue
            if truth.none_needed:
                stats["turns_none"] += 1

            short = _short_state(turn)
            long_state = turn.long_state()

            def emit(skill: str, target: str, source: str, when: str = "turn_start", evidence: str = ""):
                for variant in ("bare", "desc"):
                    for kind, state in (("short", short), ("long", long_state)):
                        examples.append(Example(
                            state=state, state_kind=kind, variant=variant,
                            skill=skill, target=target, source=source, split=split,
                            session=session, when=when, evidence=evidence,
                        ))

            for s in sorted(truth.positives):
                when = gsession.get(turn.turn_index).when.get(s, "turn_start") if gsession.get(turn.turn_index) else "turn_start"
                ev = gsession.get(turn.turn_index).evidence.get(s, "") if gsession.get(turn.turn_index) else ""
                emit(s, CORE, truth.positives[s], when, ev)
                stats["pos"] += 2
            for s, ev in sorted(truth.tangentials.items()):
                emit(s, TANGENTIAL, "distilled", evidence=ev)
                stats["tang"] += 2

            taken = set(truth.positives) | set(truth.tangentials)
            pool = sorted(set(catalog) - taken)
            n_neg = NEG_PER_POSITIVE_TURN if truth.positives else NEG_FLOOR_NONE_NEEDED
            for s in rng.sample(pool, min(n_neg, len(pool))):
                emit(s, UNRELATED, "sampled")
                stats["neg"] += 2

            stats["turns_pos"] += 1 if truth.positives else 0

    rng.shuffle(examples)
    return examples, stats


def write_jsonl(examples: list[Example], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def load_examples(path: Path) -> list[Example]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(Example(**d))
    return out


def load_golden(distilled_dir: Path) -> dict:
    """Read every distiller cache file -> {session: {turn_index: GoldenTurn}}.
    Missing dir or corrupt files simply yield fewer sessions (partial
    distillation is a valid input everywhere)."""
    from distiller import GoldenTurn

    golden: dict = {}
    distilled = Path(distilled_dir)
    if not distilled.is_dir():
        return golden
    for path in sorted(distilled.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            golden[path.stem] = {
                int(k): GoldenTurn(**v) for k, v in data.get("turns", {}).items()
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return golden
