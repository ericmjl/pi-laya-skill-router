"""Mine pi session logs into a skill-routing eval dataset.

Ground truth: pi's system prompt lists every skill; the main model loads a
skill by calling the `read` tool on its SKILL.md. We label each user turn
with the set of skills read between it and the next user message.

Output: eval/dataset.jsonl, one line per labeled turn:
  {session, ts, state, labels: [skill names], n_tool_calls}

Only `read`-tool SKILL.md loads count as labels (the sanctioned skill-load
action per pi's system prompt). Reads via bash cat/head are recorded
separately as `bash_labels` for diagnostics but not used as primary labels.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

SESSIONS = Path.home() / ".pi" / "agent" / "sessions"
OUT = Path(__file__).resolve().parent.parent / "eval" / "dataset.jsonl"

SKILL_MD_RE = re.compile(r"(SKILL\.md)$")


def text_of(content) -> str:
    """User message content -> plain text."""
    if isinstance(content, str):
        return content
    parts = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
    return "\n".join(parts)


def iter_skill_reads(msg: dict):
    """Yield (tool_name, path, via) for SKILL.md accesses in an assistant message."""
    content = msg.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "toolCall":
            continue
        name = block.get("name") or ""
        inp = block.get("input") or block.get("arguments") or {}
        if not isinstance(inp, dict):
            continue
        if name == "read":
            path = inp.get("path") or ""
            if SKILL_MD_RE.search(path.replace("\\", "/")):
                yield path, "read"
        elif name == "bash":
            cmd = inp.get("command") or ""
            for m in re.finditer(r"[\w./~-]*/skills/[\w.-]+/SKILL\.md", cmd):
                yield m.group(0), "bash"


def main() -> None:
    turns = []
    n_sessions = 0
    n_msgs = 0
    for session_file in sorted(SESSIONS.rglob("*.jsonl")):
        session_id = session_file.stem
        cwd = ""
        current: dict | None = None  # {ts, state, labels set, bash set, n_tools}

        def close():
            nonlocal current
            if current and current["state"].strip():
                turns.append(current)
            current = None

        try:
            lines = session_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = entry.get("type")
            if etype == "session":
                cwd = entry.get("cwd", "")
                continue
            if etype != "message":
                continue
            msg = entry.get("message") or {}
            role = msg.get("role")
            n_msgs += 1
            if role == "user":
                close()
                current = {
                    "session": session_id,
                    "cwd": cwd,
                    "ts": entry.get("timestamp", ""),
                    "state": text_of(msg.get("content"))[:2000],
                    "labels": set(),
                    "bash": set(),
                    "n_tool_calls": 0,
                }
            elif role == "assistant" and current is not None:
                for path, via in iter_skill_reads(msg):
                    m = re.search(r"/skills/([\w.-]+)/SKILL\.md$", path.replace("\\", "/"))
                    skill = m.group(1) if m else Path(path).parent.name
                    (current["labels"] if via == "read" else current["bash"]).add(skill)
                current["n_tool_calls"] += sum(
                    1
                    for b in (msg.get("content") or [])
                    if isinstance(b, dict) and b.get("type") == "toolCall"
                )
        close()
        n_sessions += 1

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for t in turns:
            f.write(
                json.dumps(
                    {
                        "session": t["session"],
                        "cwd": t["cwd"],
                        "ts": t["ts"],
                        "state": t["state"],
                        "labels": sorted(t["labels"]),
                        "bash_labels": sorted(t["bash"]),
                        "n_tool_calls": t["n_tool_calls"],
                    },                )
                + "\n"
            )
    read_turns = sum(1 for t in turns if t["labels"])
    print(f"sessions scanned: {n_sessions}")
    print(f"messages scanned: {n_msgs}")
    print(f"turns with skill loads: {read_turns}")
    print(f"total turns written:   {len(turns)} -> {OUT}")
    # quick diagnostics
    per_skill = defaultdict(int)
    for t in turns:
        for s in t["labels"]:
            per_skill[s] += 1
    print("\ntop read-loaded skills:")
    for s, n in sorted(per_skill.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:3d}  {s}")
    multi = [t for t in turns if len(t["labels"]) >= 2]
    print(f"\nturns loading >=2 skills via read: {len(multi)}")


if __name__ == "__main__":
    main()
