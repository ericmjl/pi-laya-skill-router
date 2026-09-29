"""Where the router's learned state lives — always outside the source repo.

The source repo is public by construction: it ships code only. Every
runtime artifact — skill catalog, distilled golden set, loop history,
mined datasets, training rows, checkpoints, result reports — is written
under a single data root that defaults to `~/.pi/agent/laya-router/data`
(override with LAYA_DATA_DIR). The data root is its own local-only git
repository so the ever-expanding golden set stays versioned without any
code history carrying data.
"""

import os
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".pi" / "agent" / "laya-router" / "data"


def data_root() -> Path:
    return Path(os.environ.get("LAYA_DATA_DIR", DEFAULT_ROOT))


def data_dir(*parts: str) -> Path:
    """Ensure and return a subdirectory of the data root."""
    p = data_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_file(*parts: str) -> Path:
    """Path (not created) for a file under the data root."""
    return data_root().joinpath(*parts)


def ensure_git() -> None:
    """Make the data root a local-only git repo (no remote, ever)."""
    root = data_root()
    if (root / ".git").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text(
        "# local-only data store: this repo must never gain a remote\n"
        "finetune/checkpoints/\n.venv/\n__pycache__/\n*.pyc\n.DS_Store\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "data: initialize laya data store"],
                   cwd=root, check=True)
