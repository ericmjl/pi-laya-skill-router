"""Laya skill-router sidecar for pi.

Loads a convaiinnovations/laya checkpoint once, then scores skill
relevance for prompts over HTTP. One POST /route call = one batched
forward pass over all skills.

Endpoints:
  GET  /health -> {loaded, model, device, skills_last}
  POST /route  -> {picks: [{name, p}], all: [{name, p}], latency_ms}
"""

import os

# Must be set before transformers/laya import; TF probe can deadlock model
# construction (learned in the laya-demo session).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
import threading
from contextlib import asynccontextmanager
from typing import Optional

import laya
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
PORT = int(os.environ.get("LAYA_PORT", "8787"))
STATE_MAX_CHARS = int(os.environ.get("LAYA_STATE_MAX_CHARS", "700"))

_state = {"agent": None, "device": None}
_predict_lock = threading.Lock()  # MPS hard-crashes on concurrent encodes


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = _pick_device()
    print(f"[laya-sidecar] loading {MODEL_ID} on {device} ...", flush=True)
    t0 = time.time()
    try:
        _state["agent"] = laya.load(MODEL_ID, device=device)
    except TypeError:
        # older SDK without device kwarg
        _state["agent"] = laya.load(MODEL_ID)
    print(f"[laya-sidecar] loaded in {time.time() - t0:.1f}s", flush=True)
    yield
    _state["agent"] = None


app = FastAPI(title="laya-skill-router", lifespan=lifespan)


class Skill(BaseModel):
    name: str
    description: str = ""


class RouteRequest(BaseModel):
    state: str
    skills: list[Skill]
    threshold: float = 0.5
    top_k: int = 3
    use_desc: bool = False
    max_picks: Optional[int] = Field(default=None, deprecated=True)


class Pick(BaseModel):
    name: str
    p: float


class RouteResponse(BaseModel):
    picks: list[Pick]
    all: list[Pick]
    latency_ms: int
    model: str
    n_scored: int


def _build_question(skill: Skill, use_desc: bool = False) -> dict:
    desc = (skill.description or "").strip()
    if len(desc) > 150:
        desc = desc[:147] + "..."
    core = "the request is exactly what this skill exists for"
    if use_desc and desc:
        core += f": {desc}"
    return {
        "type": "choice",
        "instructions": f"How related is the skill '{skill.name}' to this request?",
        "criteria": {
            "core": core,
            "tangential": "somewhat related but the request does not really need it",
            "unrelated": "no meaningful connection to the request",
        },
    }


@app.get("/health")
def health():
    return {
        "loaded": _state["agent"] is not None,
        "model": MODEL_ID,
        "device": _state["device"],
    }


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    agent = _state["agent"]
    if agent is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.skills:
        return RouteResponse(picks=[], all=[], latency_ms=0, model=MODEL_ID, n_scored=0)

    state = req.state if len(req.state) <= STATE_MAX_CHARS else req.state[: STATE_MAX_CHARS - 3] + "..."
    names = [s.name for s in req.skills]
    questions = {s.name: _build_question(s, req.use_desc) for s in req.skills}

    t0 = time.time()
    try:
        with _predict_lock:
            result = agent.predict(state, questions)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"predict failed: {exc}") from exc
    latency_ms = int((time.time() - t0) * 1000)

    scored = []
    for name in names:
        ans = result["answers"].get(name, {})
        probs = ans.get("probabilities") or {}
        scored.append(Pick(name=name, p=float(probs.get("core", 0.0))))

    scored.sort(key=lambda x: x.p, reverse=True)
    picks = [s for s in scored if s.p >= req.threshold][: req.top_k]
    return RouteResponse(
        picks=picks,
        all=scored,
        latency_ms=latency_ms,
        model=MODEL_ID,
        n_scored=len(scored),
    )


if __name__ == "__main__":
    import uvicorn

    _state["device"] = _pick_device()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
