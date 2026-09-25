"""Golden-path distillation: a frontier model labels every pi session turn
with the skills that were or should have been used.

Why a second model pass: ground truth mined from `read` calls is doubly
incomplete — skills materialize mid-turn driven by tool results (so turn-start
labels undercount), and skills the main model *should* have used but didn't
never appear at all. This module sends each session's compacted transcript to
a frontier model together with the full skill catalog and gets back, per turn:

    loaded       skills the transcript shows being used (anchors the labeler)
    should_have  skills the turn's work needed, with evidence citations
    tangential   near-misses worth training as a separate class
    when         turn_start | after_step_N — earliest point the need was
                 inferable; mid-turn `when` becomes long-context training states

Deep-module contract:
    distill_turns(turns, catalog, out_dir, ...) -> DistillReport

Hides: prompt policy, claude CLI plumbing, retries, JSON extraction,
schema validation, per-session caching/resume. Errors are defined out of
existence downstream: unknown skill names are dropped (and counted), turns
that never appear in a response default to none-needed, and failed calls
mark the session failed in the report instead of raising — a partial
distillation is always a valid one.
"""

import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from router_q import rubric_prose
from session_data import Turn, render_windows, Window

DEFAULT_CLAUDE = "claude"
DEFAULT_MODEL = "claude-sonnet-4-5"
VALID_STATUS = ("loaded", "should_have", "tangential")
WHEN_RE = re.compile(r"^(turn_start|after_step_\d+)$")


@dataclass
class GoldenTurn:
    turn_index: int
    loaded: list[str] = field(default_factory=list)
    should_have: list[str] = field(default_factory=list)
    tangential: list[str] = field(default_factory=list)
    when: dict[str, str] = field(default_factory=dict)     # skill -> turn_start|after_step_N
    evidence: dict[str, str] = field(default_factory=dict)  # skill -> citation
    confidence: str = "medium"

    @property
    def positives(self) -> list[str]:
        return self.loaded + self.should_have

    def is_empty(self) -> bool:
        return not (self.loaded or self.should_have or self.tangential)


@dataclass
class DistillReport:
    sessions_total: int = 0
    sessions_ok: int = 0
    sessions_failed: int = 0
    turns_labeled: int = 0
    turns_none_needed: int = 0
    calls: int = 0
    retries: int = 0
    dropped_skill_mentions: int = 0
    unknown_names: Counter = field(default_factory=Counter)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"sessions: {self.sessions_ok}/{self.sessions_total} ok, {self.sessions_failed} failed",
            f"turns labeled: {self.turns_labeled} (none-needed: {self.turns_none_needed})",
            f"claude calls: {self.calls} (+{self.retries} retries)",
            f"dropped unknown skill mentions: {self.dropped_skill_mentions}",
        ]
        if self.unknown_names:
            top = ", ".join(f"{k}({v})" for k, v in self.unknown_names.most_common(8))
            lines.append(f"unknown names seen: {top}")
        return "\n".join(lines)


def _catalog_block(skills: list[dict]) -> str:
    lines = []
    for s in skills:
        desc = (s.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:157] + "..."
        lines.append(f"- {s['name']}: {desc}")
    return "\n".join(lines)


def _prompt(window: Window, catalog_block: str) -> str:
    return f"""You are labeling training data for a skill-router in a coding agent (pi).
The agent has a catalog of skills. During real work it loads a skill by reading
its SKILL.md. We are building ground truth: for each user turn below, which
skills WERE used, and which SHOULD have been used?

The transcript lines are prefixed with turn numbers. Lines marked
[SKILL-READ: name] show a skill whose SKILL.md was actually read at that step —
those are confirmed `loaded` for that turn (you do not need to re-derive them).
Tool calls and their results are shown so you can see what the work actually
required.

The skill catalog (name: description):
{catalog_block}

For every turn, decide:
1. loaded — confirmed by [SKILL-READ] annotations.
2. should_have — the turn's work needed this skill, whether or not it was ever
   read. Judge by what the user asked and what the tools did. Cite the specific
   evidence (a phrase from the user, or the tool step that made the need
   concrete).
3. tangential — related but not genuinely needed.
4. none_needed — no skill in the catalog was needed. Say so explicitly.

For each skill also give `when`: "turn_start" if the need is inferable from the
user message alone, or "after_step_N" if it only became clear after tool step N.
Base `when` on the earliest point the evidence appears.

Rate each turn's overall labeling `confidence`: high / medium / low.
Use low when the transcript is too truncated to judge — those turns are dropped
from training, so do not force a label.

Skills outside the catalog do not exist; never invent names.

Answer with STRICT JSON only — no prose, no markdown fences:

{{"turns": [{{"turn": <turn number>, "confidence": "high|medium|low",
  "skills": [{{"name": "<catalog name>", "status": "loaded|should_have|tangential",
              "when": "turn_start|after_step_N", "evidence": "<short citation>"}}]}}]}}

Include EVERY turn in your answer (use an empty skills list for none-needed turns).

Transcript:
{window.text}
"""


def _extract_json(text: str) -> Optional[dict]:
    """Pull the outermost JSON object out of a possibly noisy reply."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _call_claude(prompt: str, model: str, claude_bin: str, timeout_s: int) -> str:
    proc = subprocess.run(
        [claude_bin, "-p", prompt, "--model", model, "--output-format", "json"],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exit {proc.returncode}: {proc.stderr[:300]}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude wrapper not json: {proc.stdout[:200]}")
    result = payload.get("result")
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError("claude returned empty result")
    return result


def _validate(response_text: str, catalog: set[str], report: DistillReport,
              window: Window) -> dict[int, GoldenTurn]:
    data = _extract_json(response_text)
    if not isinstance(data, dict) or not isinstance(data.get("turns"), list):
        raise ValueError("no turns array in reply")
    golden: dict[int, GoldenTurn] = {}
    for entry in data["turns"]:
        if not isinstance(entry, dict):
            continue
        try:
            turn_index = int(entry.get("turn"))
        except (TypeError, ValueError):
            continue
        if not (window.turn_range[0] <= turn_index <= window.turn_range[1]):
            continue  # hallucinated turn id from another window
        conf = str(entry.get("confidence", "medium")).lower()
        gt = GoldenTurn(turn_index=turn_index, confidence=conf if conf in ("high", "medium", "low") else "medium")
        for sk in entry.get("skills") or []:
            if not isinstance(sk, dict):
                continue
            name = str(sk.get("name", "")).strip()
            if name not in catalog:
                report.dropped_skill_mentions += 1
                report.unknown_names[name] += 1
                continue
            status = str(sk.get("status", "")).lower()
            if status not in VALID_STATUS:
                continue
            ev = str(sk.get("evidence", "")).strip()
            if status == "should_have" and not ev:
                continue  # an uncited should-have is not trainable signal
            when = str(sk.get("when", "turn_start")).strip().lower()
            if not WHEN_RE.match(when):
                when = "turn_start"
            bucket = getattr(gt, {"loaded": "loaded", "should_have": "should_have", "tangential": "tangential"}[status])
            if name in bucket:
                continue
            bucket.append(name)
            gt.when[name] = when
            gt.evidence[name] = ev[:300]
        golden[turn_index] = gt
    return golden


def _distill_window(window: Window, skills: list[dict], catalog: set[str], model: str,
                    claude_bin: str, timeout_s: int, report: DistillReport) -> dict[int, GoldenTurn]:
    prompt = _prompt(window, _catalog_block(skills))
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            report.calls += 1
            reply = _call_claude(prompt, model, claude_bin, timeout_s)
            if attempt:
                report.retries += attempt
            return _validate(reply, catalog, report, window)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            last_err = exc
    raise RuntimeError(f"window {window.session}[{window.turn_range}] failed: {last_err}")


def distill_turns(
    turns: list[Turn],
    skills: list[dict],
    out_dir: Path,
    model: str = DEFAULT_MODEL,
    claude_bin: str = DEFAULT_CLAUDE,
    budget_chars: int = 300_000,
    timeout_s: int = 900,
    workers: int = 4,
    force: bool = False,
) -> tuple[dict[str, dict[int, GoldenTurn]], DistillReport]:
    """Label every turn. Cached per session in out_dir/<session>.json; pass
    force=True to relabel. Returns {session: {turn_index: GoldenTurn}}.

    Sessions whose labeling call fails end up absent from the returned dict
    and listed in the report — callers treat missing sessions as unlabeled
    rather than treating the run as failed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = {s["name"] for s in skills}
    windows = render_windows(turns, budget_chars)
    by_session: dict[str, list[Window]] = {}
    for w in windows:
        by_session.setdefault(w.session, []).append(w)

    report = DistillReport(sessions_total=len(by_session))
    results: dict[str, dict[int, GoldenTurn]] = {}

    def load_cached(path: Path) -> Optional[dict]:
        try:
            data = json.loads(path.read_text())
            if data.get("model") == model and isinstance(data.get("turns"), dict):
                return {int(k): GoldenTurn(**v) for k, v in data["turns"].items()}
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return None

    def run_session(session: str, ses_windows: list[Window]) -> tuple[str, dict[int, GoldenTurn]]:
        cache = out_dir / f"{session}.json"
        if not force:
            cached = load_cached(cache)
            if cached is not None:
                return session, cached
        merged: dict[int, GoldenTurn] = {}
        for w in ses_windows:
            merged.update(_distill_window(w, skills, catalog, model, claude_bin, timeout_s, report))
        payload = {
            "model": model,
            "turns": {
                str(k): {
                    "turn_index": v.turn_index, "loaded": v.loaded, "should_have": v.should_have,
                    "tangential": v.tangential, "when": v.when, "evidence": v.evidence,
                    "confidence": v.confidence,
                }
                for k, v in merged.items()
            },
        }
        cache.write_text(json.dumps(payload, indent=1))
        return session, merged

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_session, s, ws): s for s, ws in by_session.items()}
        for fut in as_completed(futures):
            session = futures[fut]
            try:
                sess, golden = fut.result()
                results[sess] = golden
                report.sessions_ok += 1
            except RuntimeError as exc:
                report.sessions_failed += 1
                report.failures.append(str(exc))

    for session, golden in results.items():
        report.turns_labeled += len(golden)
        report.turns_none_needed += sum(1 for g in golden.values() if g.is_empty())
    return results, report
