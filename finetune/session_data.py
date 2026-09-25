"""pi session logs -> labeled turns and renderable transcript windows.

Deep-module contract:
    load_turns(sessions_dir) -> list[Turn]       # every user turn, all sessions
    render_windows(turns, budget_chars) -> list[Window]   # labeler-ready transcripts

Everything about the on-disk JSONL shape (message/block types, tool-call
argument names, timestamp ordering) is hidden here. Downstream modules —
the distiller, the dataset builder — only ever see Turn records and plain
text windows.
"""

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_SESSIONS_DIR = Path.home() / ".pi" / "agent" / "sessions"

SKILL_MD_RE = re.compile(r"(SKILL\.md)$")
SKILL_NAME_RE = re.compile(r"/skills/([\w.-]+)/SKILL\.md$")

# Per-entry character caps for rendered windows. Tool results are the signal
# that drives mid-turn skill needs, so results get room; assistant prose and
# images get little or none.
CAPS = {
    "user": 800,
    "assistant": 400,
    "thinking": 0,      # internal monologue: excluded entirely
    "tool_call": 200,
    "tool_result": 250,
}


@dataclass
class Event:
    """One transcript line inside a turn, in chronological order."""

    kind: str            # user | assistant | tool_call | tool_result | skill_read
    step: int            # tool-call ordinal within the turn; -1 for text lines
    text: str
    skill: Optional[str] = None  # set for skill_read events


@dataclass
class Turn:
    session: str
    ts: str
    turn_index: int              # ordinal user turn within its session
    state: str                   # the user prompt (production router's state)
    prev_state: str = ""         # previous user prompt in the same session
    observed_labels: list[str] = field(default_factory=list)  # skills read via `read`
    bash_labels: list[str] = field(default_factory=list)      # reads via bash (diagnostics)
    n_tool_calls: int = 0
    events: list[Event] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.session}#{self.turn_index}"

    def long_state(self, max_chars: int = 6000) -> str:
        """Richer state: observed reads + tool activity + prompt.

        This is the context-window training variant: what a mid-turn router
        would legitimately see. Rendered from events, not raw JSONL.
        """
        lines = [f"request: {self.state}"]
        for e in self.events:
            if e.kind == "user" and e.text.strip() == self.state.strip():
                continue
            if e.kind == "skill_read":
                lines.append(f"loaded skill {e.skill}")
            elif e.kind == "tool_call":
                lines.append(f"step {e.step} tool: {e.text}")
            elif e.kind == "tool_result":
                lines.append(f"step {e.step} result: {e.text}")
            elif e.kind == "assistant":
                lines.append(f"agent: {e.text}")
        out = "\n".join(lines)
        return out[:max_chars]


@dataclass
class Window:
    """A renderable chunk of one session, sized for one labeling call."""

    session: str
    turn_range: tuple[int, int]   # inclusive [i0, i1] of turn_index values
    text: str


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
    return "\n".join(parts)


def _skill_reads(msg: dict):
    """Yield (skill_name, via) for SKILL.md accesses in an assistant message."""
    content = msg.get("content")
    if not isinstance(content, list):
        return
    for b in content:
        if not isinstance(b, dict) or b.get("type") != "toolCall":
            continue
        name = b.get("name") or ""
        inp = b.get("input") or b.get("arguments") or {}
        if not isinstance(inp, dict):
            continue
        if name == "read":
            path = str(inp.get("path") or "")
            if SKILL_MD_RE.search(path.replace("\\", "/")):
                m = SKILL_NAME_RE.search(path.replace("\\", "/"))
                yield (m.group(1) if m else Path(path).parent.name), "read"
        elif name == "bash":
            cmd = str(inp.get("command") or "")
            for m in re.finditer(r"[\w./~-]*/skills/([\w.-]+)/SKILL\.md", cmd):
                yield m.group(1), "bash"


def _tool_call_text(b: dict) -> tuple[str, str]:
    name = b.get("name") or ""
    inp = b.get("input") or b.get("arguments") or {}
    if not isinstance(inp, dict):
        inp = {}
    if name == "bash":
        arg = str(inp.get("command", ""))[: CAPS["tool_call"]]
    elif name == "read":
        arg = str(inp.get("path", ""))[: CAPS["tool_call"]]
    else:
        arg = json.dumps(inp, ensure_ascii=False)[:CAPS["tool_call"]]
    return name, arg


def _parse_session_file(path: Path) -> list[Turn]:
    """All labeled turns in one session file, in order. OSError-proof."""
    session_id = path.stem
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    turns: list[Turn] = []
    current: Optional[Turn] = None
    step = 0  # tool-call ordinal within the open turn
    call_step: dict[str, int] = {}  # toolCall id -> step, for result attribution

    def close():
        if current is not None and current.state.strip():
            turns.append(current)

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "message":
            continue
        msg = entry.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")
        ts = entry.get("timestamp", "")

        if role == "user" and isinstance(content, str) or (
            role == "user" and isinstance(content, list)
            and any(isinstance(b, dict) and b.get("type") == "text" for b in content)
        ):
            close()
            step = 0
            current = Turn(
                session=session_id,
                ts=ts,
                turn_index=len(turns),
                state=_text_of(content)[:2000],
                prev_state=(turns[-1].state if turns else ""),
                events=[Event("user", -1, _text_of(content)[:CAPS["user"]])],
            )
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if current is None:
                    continue
                if btype == "toolCall":
                    name, arg = _tool_call_text(b)
                    current.n_tool_calls += 1
                    step += 1
                    call_id = str(b.get("id") or "")
                    if call_id:
                        call_step[call_id] = step
                    current.events.append(Event("tool_call", step, f"{name}: {arg}"))
                    for skill, via in _skill_reads({"content": [b]}):
                        if via == "read":
                            current.observed_labels.append(skill)
                            current.events.append(Event("skill_read", step, "", skill=skill))
                        else:
                            current.bash_labels.append(skill)
                elif btype == "text" and role == "assistant":
                    current.events.append(
                        Event("assistant", -1, _text_of([b])[:CAPS["assistant"]])
                    )
        elif role == "toolResult" and current is not None:
            # Results are separate messages keyed by toolCallId. Only text is
            # signal for routing; images are skipped entirely.
            rstep = call_step.get(str(entry.get("toolCallId") or ""), step)
            tool = entry.get("toolName") or "tool"
            for b in content if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "text":
                    body = str(b.get("text", ""))[:CAPS["tool_result"]].replace("\n", " ")
                    if body:
                        current.events.append(Event("tool_result", rstep, f"({tool}) {body}"))
    close()
    return turns


def load_turns(sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> list[Turn]:
    """Every user turn across every session, sessions sorted, turns ordered."""
    turns: list[Turn] = []
    for path in sorted(sessions_dir.rglob("*.jsonl")):
        turns.extend(_parse_session_file(path))
    return turns


def render_turn(turn: Turn, include_results: bool = True) -> str:
    """One turn as labeler-readable transcript lines."""
    out = [f'=== TURN {turn.turn_index} [{turn.ts[:19]}] user: "{turn.state}"']
    for e in turn.events:
        if e.kind == "user":
            continue
        if e.kind == "skill_read":
            out.append(f"    [SKILL-READ: {e.skill}]")
        elif e.kind == "tool_call":
            out.append(f"    [step {e.step}] tool: {e.text}")
        elif e.kind == "tool_result" and include_results:
            out.append(f"    [step {e.step}] result: {e.text}")
        elif e.kind == "assistant":
            out.append(f"    [agent] {e.text}")
    return "\n".join(out)


def render_windows(turns: list[Turn], budget_chars: int = 300_000) -> list[Window]:
    """Group consecutive turns into labeler-sized windows.

    A window is one distiller call. Turns are never split across windows,
    so every turn appears exactly once with full local context.
    """
    windows: list[Window] = []
    by_session: dict[str, list[Turn]] = {}
    for t in turns:
        by_session.setdefault(t.session, []).append(t)

    for session, sturns in by_session.items():
        chunks: list[list[Turn]] = []
        chunk: list[Turn] = []
        size = 0
        for t in sturns:
            tlen = len(render_turn(t))
            if chunk and size + tlen > budget_chars:
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append(t)
            size += tlen
        if chunk:
            chunks.append(chunk)
        for ch in chunks:
            text = "\n".join(render_turn(t) for t in ch)
            windows.append(
                Window(session=session, turn_range=(ch[0].turn_index, ch[-1].turn_index), text=text)
            )
    return windows


def session_split(session_id: str, splits=(0.7, 0.15, 0.15)) -> str:
    """Stable session-level split: hash -> train/dev/test. No turn leakage."""
    h = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < splits[0]:
        return "train"
    if h < splits[0] + splits[1]:
        return "dev"
    return "test"
