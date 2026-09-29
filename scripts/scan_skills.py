"""Scan all pi/agent skill directories and emit skills.json.

Collects name + description from SKILL.md YAML frontmatter for:
  ~/.agents/skills/
  ~/.pi/agent/skills/
  <project>/.agents/skills/ and <project>/.claude/skills/ (if run inside a project)

Usage: uv run scripts/scan_skills.py [--out skills.json] [--project <cwd>]
"""

import argparse
import json
import os
from pathlib import Path

import yaml

GLOBAL_DIRS = [
    Path.home() / ".agents" / "skills",
    Path.home() / ".pi" / "agent" / "skills",
]
PROJECT_SUBDIRS = [".agents/skills", ".claude/skills"]


def parse_frontmatter(skill_md: Path) -> dict:
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}
    return meta if isinstance(meta, dict) else {}


def scan(project: str | None) -> list[dict]:
    dirs = list(GLOBAL_DIRS)
    if project:
        for sub in PROJECT_SUBDIRS:
            dirs.append(Path(project) / sub)
    skills = []
    seen_names = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for skill_md in sorted(d.glob("*/SKILL.md")):
            meta = parse_frontmatter(skill_md)
            name = meta.get("name") or skill_md.parent.name
            if name in seen_names:
                continue
            seen_names.add(name)
            skills.append(
                {
                    "name": name,
                    "description": meta.get("description", ""),
                    "path": str(skill_md),
                    "source": str(d),
                }
            )
    return skills


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--project", default=os.getcwd())
    args = ap.parse_args()

    skills = scan(args.project)
    if args.out:
        out = Path(args.out)
    else:
        from data_paths import data_file
        out = data_file("skills.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(skills, indent=2) + "\n")
    print(f"{len(skills)} skills -> {out}")
    by_src: dict[str, int] = {}
    for s in skills:
        by_src[s["source"]] = by_src.get(s["source"], 0) + 1
    for src, n in sorted(by_src.items()):
        print(f"  {n:3d}  {src}")


if __name__ == "__main__":
    main()
