"""Mine the router's own picks out of ~/.pi/agent/laya-router/log.jsonl and
join them to the turns they were made on.

Why: uniform random sampling throws one-in-N negatives at the router, so the
skills it confidently over-picks in production (the real mistakes) appear in
training no more often than coin-flip skills. Every route log entry is an
on-distribution false positive waiting to be claimed — this module joins
those picks to turns so the dataset builder and the eval can classify them
against ground truth.

Deliberate split of responsibilities: mine_picks does ONLY the join (log
entry -> turn), which depends on nothing but the log and the session files.
Classification — is this pick a positive, picked-tangential, or true
negative — lives in dataset.py beside `turn_truth` (the single merge of
observed + golden labels), so these caches never go stale when the nightly
golden set grows; consumers reclassify against current truth every run.

Deep-module contract:
    mine_picks(turns, log_path, out_dir) -> (picks, MineReport)
    load_picks(out_dir) -> dict[turn_key, list[JoinedPick]]

Join rules:
  - Window: a route is assigned to the turn whose [ts_i - SLACK, ts_{i+1} -
    SLACK) window contains it. The slack absorbs the observed race where the
    route logs at input time a few milliseconds BEFORE the user message is
    persisted to the session file (measured: -11ms to +3.1s). A route
    belonging to turn i but logged just before turn i+1's message lands
    still attributes to i, which is what the extension's generation guard
    already guarantees semantically.
  - Identity: entries logged by the current extension carry `session_stem`
    (the session-file name, which session_data.py uses as its session id —
    exact join) and `session_id` (pi's bare uuid — fallback identity).
    Entries with neither (legacy backlog) fall back to a fuzzy join on the
    100-char `prompt_head` prefix, accepted only when the timestamp also
    lands inside the turn's window.
  - Window: route ts always lands after the turn's user message (routes fire
    on input; measured +0.0..+3.1s), so an entry belongs to the turn whose
    [ts_i, ts_{i+1}) window contains it. Multiple entries may join one turn
    (input-time prefetch re-fired when the skill set changes): picks merge
    by union, keeping the max probability per skill.
  - Unmatched entries are dropped and counted — the backlog is a bonus, not
    a requirement.
  - `already_in_context` picks are dropped here: the skill body was already
    in the transcript, so "never used" is ambiguous. Every other skip rule
    (unknown truth, positive, tangential) is classification and belongs to
    the truth-merge owner.

Errors are defined out of existence downstream, same policy as the
distiller: corrupt log lines, unparseable timestamps, and unknown skill
names are dropped and counted, never raised.
"""

import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from session_data import Turn

CACHE_VERSION = 1
WINDOW_SLACK_S = 5.0  # route ts may precede the turn's persisted ts by a few ms


@dataclass
class JoinedPick:
    """One router pick joined to its turn, pre-classification."""

    skill: str
    p: float
    ts: str                 # route log timestamp
    matched: str            # exact | fuzzy
    source: str = "router"  # provenance tag; classification renames the row


@dataclass
class MineReport:
    routes_total: int = 0
    routes_joined: int = 0
    joined_exact: int = 0
    joined_fuzzy: int = 0
    routes_dropped: int = 0          # no identity match / no turn in window / bad ts
    picks_total: int = 0
    picks_in_context: int = 0        # already_in_context: ambiguous, dropped here
    picks_merged: int = 0            # duplicate pick of one skill in one turn
    turns_with_picks: int = 0
    sessions_cached: int = 0
    sessions_updated: int = 0        # cache content changed this run

    def summary(self) -> str:
        return (
            f"routes: {self.routes_joined}/{self.routes_total} joined "
            f"({self.joined_exact} exact, {self.joined_fuzzy} fuzzy, {self.routes_dropped} dropped); "
            f"picks: {self.picks_total} ({self.picks_in_context} in-context skipped, "
            f"{self.picks_merged} duplicates merged); "
            f"{self.turns_with_picks} turns across {self.sessions_cached} sessions "
            f"({self.sessions_updated} caches updated)"
        )


def _ts(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _prefix_matches(turn: Turn, head: str) -> bool:
    """Does the logged prompt_head prefix the turn's state?

    The log stores computeState(prompt): the raw prompt, or `prev || prompt`
    when the prompt is under 60 chars. Session turns store the raw prompt,
    so both shapes must be tried.
    """
    state = turn.state.strip()
    if state.startswith(head):
        return True
    prev = turn.prev_state.strip()
    return bool(prev) and (prev[:600] + " || " + state).startswith(head)


def _turn_windows(sturns: list[Turn]) -> tuple[list[datetime], list[Turn]]:
    """Turn start timestamps + the turns themselves, in order."""
    starts, kept = [], []
    for t in sturns:
        ts = _ts(t.ts)
        if ts is not None:
            starts.append(ts)
            kept.append(t)
    return starts, kept


def _window_turn(starts: list[datetime], kept: list[Turn], route_ts: datetime) -> Optional[Turn]:
    """The turn whose [ts_i - SLACK, ts_{i+1} - SLACK) window contains route_ts."""
    slack = timedelta(seconds=WINDOW_SLACK_S)
    i = bisect_right([s - slack for s in starts], route_ts) - 1
    return kept[i] if i >= 0 else None


def mine_picks(
    turns: list[Turn],
    log_path: Path,
    out_dir: Path,
) -> tuple[dict[str, list[JoinedPick]], MineReport]:
    """Join every route log entry to its turn.

    Returns ({turn_key: [JoinedPick]}, MineReport). Cached per session in
    out_dir/<session>.json; a cache is rewritten only when its content
    changes, so callers can use sessions_updated to detect new signal.
    """
    rep = MineReport()
    by_session: dict[str, list[Turn]] = {}
    for t in turns:
        by_session.setdefault(t.session, []).append(t)
    windows = {s: _turn_windows(sorted(sturns, key=lambda t: t.ts))
               for s, sturns in by_session.items()}

    entries: list[dict] = []
    try:
        for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "route":
                entries.append(e)
    except OSError:
        return {}, rep
    rep.routes_total = len(entries)

    picks: dict[str, dict[str, JoinedPick]] = {}  # turn_key -> skill -> pick

    def join(entry: dict, turn: Turn, matched: str) -> None:
        in_ctx = set(entry.get("already_in_context") or [])
        for p in entry.get("picks") or []:
            name = str(p.get("name", ""))
            if not name:
                continue
            rep.picks_total += 1
            if name in in_ctx:
                rep.picks_in_context += 1
                continue
            existing = picks.setdefault(turn.key, {})
            if name in existing:
                rep.picks_merged += 1
                existing[name].p = max(existing[name].p, float(p.get("p", 0.0)))
                continue
            existing[name] = JoinedPick(
                skill=name, p=float(p.get("p", 0.0)),
                ts=str(entry.get("ts", "")), matched=matched,
            )

    for entry in entries:
        route_ts = _ts(entry.get("ts", ""))
        if route_ts is None:
            rep.routes_dropped += 1
            continue
        stem = entry.get("session_stem")
        ident = stem or entry.get("session_id")
        if ident and ident in windows:
            starts, kept = windows[ident]
            turn = _window_turn(starts, kept, route_ts)
            if turn is None:
                rep.routes_dropped += 1
                continue
            join(entry, turn, "exact")
            rep.joined_exact += 1
            continue
        # Fuzzy fallback (legacy entries): prefix match + same ts window.
        head = str(entry.get("prompt_head", "")).strip()
        if not head:
            rep.routes_dropped += 1
            continue
        hit = None
        for session in sorted(windows):
            starts, kept = windows[session]
            turn = _window_turn(starts, kept, route_ts)
            if turn is not None and _prefix_matches(turn, head):
                hit = turn
                break
        if hit is None:
            rep.routes_dropped += 1
            continue
        join(entry, hit, "fuzzy")
        rep.joined_fuzzy += 1

    rep.routes_joined = rep.joined_exact + rep.joined_fuzzy
    rep.turns_with_picks = len(picks)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_sess: dict[str, dict[int, JoinedPick]] = {}
    for key, sk in picks.items():
        session, _, idx = key.rpartition("#")
        by_sess.setdefault(session, {})[int(idx)] = sk
    for session, turn_picks in sorted(by_sess.items()):
        payload = {
            "version": CACHE_VERSION,
            "turns": {
                str(idx): [asdict(v) for v in sorted(sk.values(), key=lambda v: v.skill)]
                for idx, sk in sorted(turn_picks.items())
            },
        }
        cache = out_dir / f"{session}.json"
        text = json.dumps(payload, indent=1)
        rep.sessions_cached += 1
        try:
            if cache.exists() and cache.read_text() == text:
                continue
            cache.write_text(text)
            rep.sessions_updated += 1
        except OSError:
            continue
    return (
        {key: list(sk.values()) for key, sk in picks.items()},
        rep,
    )


def load_picks(picks_dir: Path) -> dict[str, list[JoinedPick]]:
    """Read every picks cache -> {turn_key: [JoinedPick]}. Missing dir or
    corrupt files simply yield fewer sessions (partial mining is a valid
    input everywhere), matching load_golden's policy."""
    picks: dict[str, list[JoinedPick]] = {}
    picks_dir = Path(picks_dir)
    if not picks_dir.is_dir():
        return picks
    for path in sorted(picks_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            for idx, rows in data.get("turns", {}).items():
                picks[f"{path.stem}#{int(idx)}"] = [JoinedPick(**r) for r in rows]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return picks
