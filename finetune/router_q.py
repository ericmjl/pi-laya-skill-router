"""The train/serve format contract for the skill router.

Every producer of a laya question — the production sidecar, the fine-tuning
trainer, the offline eval, and the golden-path distiller's prompt — must get
its wording from this module. If the criteria text drifts between training
and serving, the checkpoint silently scores against a different rubric than
it was trained on; that failure mode is why this file exists and why nothing
else may hardcode criteria strings.
"""

from typing import Optional

# Criteria wording is byte-identical to sidecar/server.py `_build_question`
# (which predates this module; a later refactor should make the sidecar
# import from here too).
CRITERIA = {
    "core": "the request is exactly what this skill exists for",
    "tangential": "somewhat related but the request does not really need it",
    "unrelated": "no meaningful connection to the request",
}

CORE, TANGENTIAL, UNRELATED = "core", "tangential", "unrelated"
TARGET_INDEX = {CORE: 0, TANGENTIAL: 1, UNRELATED: 2}
TARGETS = (CORE, TANGENTIAL, UNRELATED)


def build_question(skill_name: str, description: str = "", use_desc: bool = False) -> dict:
    """Question definition for one skill, mirroring sidecar/server.py."""
    desc = (description or "").strip()
    if len(desc) > 150:
        desc = desc[:147] + "..."
    core = CRITERIA[CORE]
    if use_desc and desc:
        core += f": {desc}"
    return {
        "type": "choice",
        "instructions": f"How related is the skill '{skill_name}' to this request?",
        "criteria": dict(CRITERIA, core=core),
    }


def cap_desc(description: str, limit: int = 220) -> str:
    """Description trimmed to labeler/eval size."""
    d = (description or "").strip()
    return d if len(d) <= limit else d[: limit - 3] + "..."


def target_vector(target: str):
    """One-hot target over the three options, in criteria order."""
    v = [0.0, 0.0, 0.0]
    v[TARGET_INDEX[target]] = 1.0
    return v


def rubric_prose() -> str:
    """The criteria as prose for the distiller's prompt — same wording, so the
    golden labels judge skills by the exact rubric laya is scored on."""
    return "\n".join(f"  - {k}: {v}" for k, v in CRITERIA.items())
