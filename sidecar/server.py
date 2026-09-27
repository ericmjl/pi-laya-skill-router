"""Laya skill-router sidecar for pi.

Loads a convaiinnovations/laya checkpoint once, then scores skill
relevance for prompts over HTTP. One POST /route call = one batched
forward pass over all skills (or a shortlist when a stage-1 bi-encoder
is configured — see docs/ROUTER_LATENCY_STRATEGY.md).

Endpoints:
  GET  /health -> {loaded, model, device, warm}
  POST /route  -> {picks: [{name, p}], all: [{name, p}], latency_ms,
                   path, cached, s1_ms, s2_ms, n_scored}

Latency work (docs/ROUTER_LATENCY_STRATEGY.md):
  - exact-match LRU cache on (state, skill-fingerprint, threshold, top_k,
    checkpoint) — predict is deterministic, so a hit returns the identical
    answer;
  - multi-shape startup warmup + optional keep-alive ping, because MPS
    kernel recompilation after idle is a large share of the production
    p50-vs-warm gap and the timeout tail;
  - optional two-stage scoring: a distilled bi-encoder (LAYA_BIENCODER)
    shortlists LAYA_SHORTLIST_K skills, laya scores only those rows.
    Without LAYA_BIENCODER set, routing is full-catalog exactly as before.
"""

import os

# Must be set before transformers/laya import; TF probe can deadlock model
# construction (learned in the laya-demo session).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import json
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

import laya
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
PORT = int(os.environ.get("LAYA_PORT", "7699"))
STATE_MAX_CHARS = int(os.environ.get("LAYA_STATE_MAX_CHARS", "700"))
CACHE_SIZE = int(os.environ.get("LAYA_CACHE_SIZE", "512"))
WARM_SHAPES = [int(n) for n in os.environ.get("LAYA_WARM_SHAPES", "1,8,16,24,110").split(",")]
KEEPALIVE_S = float(os.environ.get("LAYA_KEEPALIVE_S", "25"))  # 0 disables
BIENCODER_PATH = os.environ.get("LAYA_BIENCODER", "")  # empty = full catalog
SHORTLIST_K = int(os.environ.get("LAYA_SHORTLIST_K", "16"))  # 0 disables
S1_MIN_SIM = float(os.environ.get("LAYA_S1_MIN_SIM", "0.35"))  # below -> full catalog

_state = {"agent": None, "device": None, "warm": False}
_predict_lock = threading.Lock()  # MPS hard-crashes on concurrent encodes
_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _BiEncoder:
    """Stage-1 dual-tower ranker (finetune/train_biencoder.py output).

    Encodes the state once and each skill document once per catalog
    fingerprint; ranking is a cosine over cached vectors. Trained to
    reproduce the cross-encoder's ranking, so the shortlist it hands laya
    is checked against the harness before promotion.
    """

    def __init__(self, path: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        # train_biencoder.py saves encoder/ and tokenizer/ under the root
        self.tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        self.model = AutoModel.from_pretrained(os.path.join(path, "encoder")).to(device).eval()
        self.device = device
        self.max_chars = 700   # mirror the state cap the cross-encoder sees
        self.doc_cap = 400     # skill docs are capped the same at training
        self._fp = None
        self._doc_vecs = None  # (n, d) float32 tensor on device

    @torch.no_grad()
    def _embed(self, texts: list[str]) -> torch.Tensor:
        enc = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (out * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(vec, dim=-1)

    def score(self, state: str, skills: list) -> tuple[list[str], float, float]:
        """Return (ranked_names, max_sim, elapsed_ms)."""
        t0 = time.time()
        fp = hashlib.sha1(
            json.dumps([[s.name, s.description[: self.doc_cap]] for s in skills]).encode()
        ).hexdigest()
        if fp != self._fp:
            docs = [f"{s.name}: {s.description[: self.doc_cap]}" for s in skills]
            self._doc_vecs = self._embed(docs)
            self._fp = fp
        q = self._embed([state[: self.max_chars]])
        sims = (q @ self._doc_vecs.T)[0]
        order = torch.argsort(sims, descending=True)
        ranked = [skills[i].name for i in order.tolist()]
        return ranked, float(sims[order[0]]), (time.time() - t0) * 1000


def _load_biencoder(device: str) -> Optional[_BiEncoder]:
    if not BIENCODER_PATH:
        return None
    try:
        enc = _BiEncoder(BIENCODER_PATH, device)
        print(f"[laya-sidecar] stage-1 bi-encoder loaded from {BIENCODER_PATH}", flush=True)
        return enc
    except Exception as exc:  # noqa: BLE001 - absent/broken stage-1 must not kill serving
        print(f"[laya-sidecar] LAYA_BIENCODER set but failed to load ({exc}); "
              f"serving full catalog", flush=True)
        return None


_bi = {"encoder": None}


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
    _bi["encoder"] = _load_biencoder(device)
    _warmup()
    if KEEPALIVE_S > 0:
        t = threading.Thread(target=_keepalive, daemon=True)
        t.start()
    yield
    _state["agent"] = None


def _warm_rows(n: int) -> dict:
    """One predict call over n synthetic skills — compiles the Metal kernels
    for this batch shape so the first real route doesn't pay for them."""
    agent = _state["agent"]
    skills = {f"warmup-skill-{i}": {
        "type": "choice",
        "instructions": f"How related is the skill 'warmup-skill-{i}' to this request?",
        "criteria": {
            "core": "the request is exactly what this skill exists for",
            "tangential": "somewhat related but the request does not really need it",
            "unrelated": "no meaningful connection to the request",
        },
    } for i in range(n)}
    state = ("warm the router with a representative seven-hundred character state so the "
             "batch shape and tokenization path are compiled before any real turn needs them. " * 6)[:700]
    with _predict_lock:
        agent.predict(state, skills)


def _warmup():
    t0 = time.time()
    try:
        for n in WARM_SHAPES:
            _warm_rows(n)
        print(f"[laya-sidecar] warmup done in {time.time() - t0:.1f}s "
              f"(shapes {WARM_SHAPES})", flush=True)
        _state["warm"] = True
    except Exception as exc:  # noqa: BLE001 - warmup failure must not kill serving
        print(f"[laya-sidecar] warmup failed ({exc})", flush=True)


def _keepalive():
    while True:
        time.sleep(KEEPALIVE_S)
        try:
            _warm_rows(2)
        except Exception:  # noqa: BLE001 - a failed ping retries next cycle
            pass


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
    path: str = "full"          # full | shortlist | shortlist-fallback | cache
    cached: bool = False
    s1_ms: int = 0              # stage-1 (shortlist) time; 0 on full passes
    s2_ms: int = 0              # laya pass time
    shortlist: list[str] = []   # candidates stage-1 selected (empty on full)


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


def _skills_fp(skills: list[Skill]) -> str:
    return hashlib.sha1(
        json.dumps([[s.name, s.description] for s in skills]).encode()
    ).hexdigest()


def _score_locked(agent, state: str, skills: list[Skill], use_desc: bool = False) -> list[Pick]:
    """One laya pass over the given skills. Caller holds _predict_lock.
    Returns p(core) per skill, sorted."""
    names = [s.name for s in skills]
    questions = {s.name: _build_question(s, use_desc) for s in skills}
    result = agent.predict(state, questions)
    scored = []
    for name in names:
        ans = result["answers"].get(name, {})
        probs = ans.get("probabilities") or {}
        scored.append(Pick(name=name, p=float(probs.get("core", 0.0))))
    scored.sort(key=lambda x: x.p, reverse=True)
    return scored


@app.get("/health")
def health():
    return {
        "loaded": _state["agent"] is not None,
        "model": MODEL_ID,
        "device": _state["device"],
        "warm": _state["warm"],
        "shortlist": SHORTLIST_K if _bi["encoder"] is not None else 0,
        "cache": len(_cache),
    }


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    agent = _state["agent"]
    if agent is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.skills:
        return RouteResponse(picks=[], all=[], latency_ms=0, model=MODEL_ID, n_scored=0)

    state = req.state if len(req.state) <= STATE_MAX_CHARS else req.state[: STATE_MAX_CHARS - 3] + "..."
    # use_desc changes the question text, so it changes the answer: it must be
    # part of the cache key (a collision silently returns the wrong variant).
    key = (state, _skills_fp(req.skills), req.threshold, req.top_k, req.use_desc, MODEL_ID)

    t_lookup = time.time()
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return RouteResponse(
            picks=hit["picks"], all=hit["all"],
            latency_ms=int((time.time() - t_lookup) * 1000),
            model=MODEL_ID, n_scored=hit["n_scored"], path="cache", cached=True,
            s1_ms=0, s2_ms=hit["s2_ms"], shortlist=hit["shortlist"],
        )

    t0 = time.time()
    skills = req.skills
    s1_ms = 0
    path = "full"
    shortlist_names: list[str] = []

    # ONE lock around both stages: the bi-encoder and laya share the MPS
    # device, and concurrent GPU work from two threads crashes Metal
    # (MTLCommandBufferStatusCommitted assertion) — not just corrupts.
    try:
        with _predict_lock:
            encoder = _bi["encoder"]
            if encoder is not None and 0 < SHORTLIST_K < len(skills):
                ranked, max_sim, s1_ms = encoder.score(state, skills)
                if max_sim >= S1_MIN_SIM:
                    shortlist_names = ranked[:SHORTLIST_K]
                    keep = set(shortlist_names)
                    skills = [s for s in skills if s.name in keep]
                    path = "shortlist"
                else:
                    path = "shortlist-fallback"  # stage-1 unconfident: score everything
            scored = _score_locked(agent, state, skills, req.use_desc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"predict failed: {exc}") from exc
    s2_ms = int((time.time() - t0) * 1000) - int(s1_ms)
    latency_ms = int((time.time() - t0) * 1000)

    picks = [s for s in scored if s.p >= req.threshold][: req.top_k]
    resp = RouteResponse(
        picks=picks, all=scored, latency_ms=latency_ms, model=MODEL_ID,
        n_scored=len(scored), path=path, cached=False, s1_ms=int(s1_ms), s2_ms=s2_ms,
        shortlist=shortlist_names,
    )
    _cache[key] = {"picks": picks, "all": scored, "n_scored": len(scored),
                   "s2_ms": s2_ms, "shortlist": shortlist_names}
    _cache.move_to_end(key)
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return resp


if __name__ == "__main__":
    import uvicorn

    _state["device"] = _pick_device()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
