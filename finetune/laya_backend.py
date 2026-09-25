"""Everything that touches the laya checkpoint format or the laya SDK.

Deep-module contract:
    load_checkpoint(id_or_path) -> ModelBundle
    save_checkpoint(bundle, out_dir, max_len=None, note=None)   # laya-loadable dir
    encode_examples(bundle, examples, max_len) -> list[batch]    # training batches
    predict_ranking(bundle, state, skills, ...) -> Ranking       # production-faithful scores

Hides: build_sequence/marker internals, collation, safetensors layout,
config surgery for extended context, and the exact decode path the
production sidecar uses (predict_ranking delegates to the stock Agent so
eval numbers are computed under the same temperature/decoding rules as
serving).
"""

import copy
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os

os.environ.setdefault("USE_TF", "0")  # TF probe can deadlock model construction

import torch
from safetensors.torch import save_file

import laya
from laya.common import build_sequence, collate_items

from router_q import build_question, target_vector, CORE, TANGENTIAL, UNRELATED

TARGET_ORDER = (CORE, TANGENTIAL, UNRELATED)

DEFAULT_BASE = "convaiinnovations/laya"


@dataclass
class ModelBundle:
    agent: object                      # laya Agent (model, tok, cfg, device)
    source: str
    max_len: int
    head_max_len: int

    @property
    def device(self) -> str:
        return str(self.agent.device)


@dataclass
class Ranking:
    ranked: list[tuple[str, float]]    # [(skill, p_core)] sorted desc
    picks: list[tuple[str, float]]     # threshold + top_k applied
    latency_ms: int


def load_checkpoint(id_or_path: str = DEFAULT_BASE, device: str = "mps") -> ModelBundle:
    agent = laya.load(id_or_path, device=device)
    return ModelBundle(
        agent=agent,
        source=id_or_path,
        max_len=int(agent.cfg.get("max_len", 512)),
        head_max_len=int(agent.cfg.get("head_max_len", 192)),
    )


def save_checkpoint(bundle: ModelBundle, out_dir: Path, max_len: Optional[int] = None,
                    head_max_len: Optional[int] = None, note: str = "") -> Path:
    """Write a directory the stock laya.load() accepts, byte-for-byte the
    same layout as upstream checkpoints, with any config overrides applied."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. weights
    state = {k: v.contiguous().cpu() for k, v in bundle.agent.model.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"))

    # 2. encoder + tokenizer configs, written by their own objects so any
    #    transformers-version quirks round-trip exactly.
    bundle.agent.model.encoder.config.save_pretrained(str(out_dir / "encoder"))
    bundle.agent.tok.save_pretrained(str(out_dir / "tokenizer"))

    # 3. agent config: copy + overrides + provenance
    cfg = copy.deepcopy(bundle.agent.cfg)
    if max_len is not None:
        cfg["max_len"] = int(max_len)
    if head_max_len is not None:
        cfg["head_max_len"] = int(head_max_len)
    cfg.setdefault("training", {})
    cfg["training"].update(
        {
            "fine_tuned_from_checkpoint": True,
            "fine_tuned_from": bundle.source,
            "note": note or "skill-router SFT",
        }
    )
    (out_dir / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    return out_dir


def encode_examples(bundle: ModelBundle, examples: list[dict], max_len: int) -> list[dict]:
    """Tokenize {state, question, target} rows into collatable items.

    The sequence construction is exactly the SDK's `build_sequence` — the
    same code path inference uses — so train and serve tokenization cannot
    drift. `target` is one of router_q TARGETS. Returns flat items; the
    caller (trainer) owns batching and epoch policy and collates chunks
    with `collate_chunk`.
    """
    tok = bundle.agent.tok
    items = []
    for ex in examples:
        q = bundle.agent._to_internal(ex["question"])
        ids, markers = build_sequence(
            tok, ex["state"], q, max_len=max_len,
            head_max_len=bundle.head_max_len,
        )
        n_opts = len(q["crit"])
        assert len(markers) == n_opts, f"option count mismatch for {ex.get('skill')}"
        items.append(
            {
                "ids": ids,
                "markers": markers,
                "qtype": laya.QTYPES[q["t"]],
                "target": target_vector(ex["target"]),
                "label": TARGET_ORDER.index(ex["target"]),
                "meta": {"skill": ex.get("skill", "")},
            }
        )
    return items


def collate_chunk(items: list[dict], pad_id: int) -> Optional[dict]:
    """Collate one chunk of encoded items into a tensor batch."""
    return collate_items([items], pad_id)


def predict_ranking(
    bundle: ModelBundle,
    state: str,
    skills: list[dict],
    use_desc: bool = False,
    max_len: Optional[int] = None,
    threshold: float = 0.3,
    top_k: int = 10,
) -> Ranking:
    """Score every skill for one state through the stock Agent.decode path —
    identical to what the sidecar + pi extension see in production.

    The state is char-capped to what max_len can hold (~3 chars/token):
    build_sequence tokenizes the state once per question, so an oversized
    state costs 112x its own tokenization for tokens the model never reads.
    """
    effective_max = max_len or bundle.max_len
    state = state[: effective_max * 3]
    questions = {s["name"]: build_question(s["name"], s.get("description", ""), use_desc) for s in skills}
    t0 = time.time()
    result = bundle.agent.predict(state, questions, max_len=max_len)
    latency_ms = int((time.time() - t0) * 1000)
    scored = []
    for s in skills:
        ans = result["answers"].get(s["name"], {})
        p = float((ans.get("probabilities") or {}).get("core", 0.0))
        scored.append((s["name"], p))
    scored.sort(key=lambda x: x[1], reverse=True)
    picks = [x for x in scored if x[1] >= threshold][:top_k]
    return Ranking(ranked=scored, picks=picks, latency_ms=latency_ms)
