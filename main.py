"""
chainstate-encoder — semantic grounding service for CHAINSTATE v0.7.0

Runs sentence-transformers/all-MiniLM-L6-v2 (22M params, 384-dim output,
~90MB on disk, sub-100ms inference on CPU) behind a tiny FastAPI surface.

Every CHAINSTATE receipt can now be embedded into a real semantic vector
space alongside its 65,536-dim symbolic representation. Cosine similarity
between receipts becomes a meaningful measure of semantic proximity.

Endpoints:
  GET  /               → welcome + version
  GET  /health         → { ok: true, model, dim, cache_size }
  POST /embed          → { vector: [384 floats], model, dim, elapsed_ms }
  POST /cosine         → { cos: float, elapsed_ms }
  POST /nearest        → { neighbors: [{text, cos}], elapsed_ms }
  POST /cache/upsert   → cache a text with a label (for /nearest to return)
  GET  /cache/list     → list cached labels
  DELETE /cache/{label}→ remove a cached embedding

The cache is in-process (dict, bounded to CACHE_MAX entries with LRU eviction).
For durable priors storage, the priors ingester (piece 3) writes to Cloudflare
KV directly — this service is purely inference + ephemeral neighbor lookup.

Env:
  MODEL_NAME    → default 'sentence-transformers/all-MiniLM-L6-v2'
  CACHE_MAX     → default 10000 entries
  API_KEY       → optional bearer token; if set, all POST endpoints require it
  CORS_ORIGINS  → default '*'

Owner: Ciprian Florin Pater
"""
import os
import time
from collections import OrderedDict
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_NAME   = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_MAX    = int(os.getenv("CACHE_MAX", "10000"))
API_KEY      = os.getenv("API_KEY", "").strip()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
SERVICE_VER  = "0.7.0-encoder-2026-07-15"

# ─── Model load (once, at boot) ──────────────────────────────────────────
print(f"[{SERVICE_VER}] loading {MODEL_NAME} ...", flush=True)
t0 = time.time()
model = SentenceTransformer(MODEL_NAME)
DIM   = int(model.get_sentence_embedding_dimension())
print(f"[{SERVICE_VER}] loaded in {time.time()-t0:.1f}s · dim={DIM}", flush=True)

# ─── LRU cache for /nearest lookups ──────────────────────────────────────
# key   = label (str, chosen by caller)
# value = { "text": str, "vec": np.ndarray[float32, DIM], "ts": float }
_cache: "OrderedDict[str, dict]" = OrderedDict()

def _cache_put(label: str, text: str, vec: np.ndarray):
    if label in _cache:
        _cache.move_to_end(label)
    _cache[label] = {"text": text, "vec": vec.astype(np.float32), "ts": time.time()}
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)  # evict LRU

# ─── FastAPI setup ───────────────────────────────────────────────────────
app = FastAPI(
    title="chainstate-encoder",
    version=SERVICE_VER,
    description="Semantic grounding service for CHAINSTATE — MiniLM-L6-v2 behind FastAPI",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Service-Version", "X-Elapsed-Ms"],
)

def _auth(authorization: Optional[str] = Header(None)) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1].strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

# ─── Schemas ─────────────────────────────────────────────────────────────
class EmbedIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192)
    normalize: bool = True
    cache_as: Optional[str] = Field(None, description="Optional label to cache under for /nearest")

class EmbedOut(BaseModel):
    vector: List[float]
    model: str
    dim: int
    elapsed_ms: int
    cached_as: Optional[str] = None

class CosineIn(BaseModel):
    a: str = Field(..., min_length=1, max_length=8192)
    b: str = Field(..., min_length=1, max_length=8192)

class NearestIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192)
    k: int = Field(5, ge=1, le=50)

class CacheUpsertIn(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    text: str  = Field(..., min_length=1, max_length=8192)

# ─── Endpoints ───────────────────────────────────────────────────────────
@app.get("/")
def welcome():
    return {
        "service": "chainstate-encoder",
        "version": SERVICE_VER,
        "model": MODEL_NAME,
        "dim": DIM,
        "endpoints": [
            "GET  /                → this page",
            "GET  /health          → readiness + cache stats",
            "POST /embed           → text → 384-dim vector",
            "POST /cosine          → similarity between two texts",
            "POST /nearest         → k-nearest cached labels for a text",
            "POST /cache/upsert    → add a label+text to the neighbor pool",
            "GET  /cache/list      → labels currently cached",
            "DELETE /cache/{label} → drop a cached label",
        ],
        "owner": "Ciprian Florin Pater",
        "chainstate_worker": "https://chainstate-worker.ciprianpater.workers.dev",
    }

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "dim": DIM, "cache_size": len(_cache), "cache_max": CACHE_MAX}

@app.post("/embed", response_model=EmbedOut)
def embed(body: EmbedIn, _: None = Depends(_auth)):
    t0 = time.time()
    vec = model.encode(body.text, normalize_embeddings=body.normalize)
    if hasattr(vec, "tolist"):
        arr = np.asarray(vec, dtype=np.float32)
    else:
        arr = np.array(vec, dtype=np.float32)
    if body.cache_as:
        _cache_put(body.cache_as, body.text, arr)
    return EmbedOut(
        vector=arr.tolist(),
        model=MODEL_NAME,
        dim=DIM,
        elapsed_ms=int((time.time() - t0) * 1000),
        cached_as=body.cache_as,
    )

@app.post("/cosine")
def cosine(body: CosineIn, _: None = Depends(_auth)):
    t0 = time.time()
    v = model.encode([body.a, body.b], normalize_embeddings=True)
    va, vb = np.asarray(v[0], dtype=np.float32), np.asarray(v[1], dtype=np.float32)
    cos = float(np.dot(va, vb))  # already normalized
    return {"cos": cos, "model": MODEL_NAME, "elapsed_ms": int((time.time() - t0) * 1000)}

@app.post("/nearest")
def nearest(body: NearestIn, _: None = Depends(_auth)):
    if not _cache:
        return {"neighbors": [], "note": "cache empty — POST /cache/upsert first", "elapsed_ms": 0}
    t0 = time.time()
    q = np.asarray(model.encode(body.text, normalize_embeddings=True), dtype=np.float32)
    labels = list(_cache.keys())
    mat = np.stack([_cache[k]["vec"] for k in labels])
    # both are unit-norm → dot product = cosine similarity
    sims = mat @ q
    order = np.argsort(-sims)[: body.k]
    neighbors = []
    for i in order:
        neighbors.append({
            "label": labels[int(i)],
            "text": _cache[labels[int(i)]]["text"][:400],
            "cos": float(sims[int(i)]),
        })
    return {"neighbors": neighbors, "cache_size": len(_cache), "elapsed_ms": int((time.time() - t0) * 1000)}

@app.post("/cache/upsert")
def cache_upsert(body: CacheUpsertIn, _: None = Depends(_auth)):
    t0 = time.time()
    v = np.asarray(model.encode(body.text, normalize_embeddings=True), dtype=np.float32)
    _cache_put(body.label, body.text, v)
    return {"ok": True, "label": body.label, "cache_size": len(_cache), "elapsed_ms": int((time.time() - t0) * 1000)}

@app.get("/cache/list")
def cache_list():
    return {"count": len(_cache), "labels": list(_cache.keys())[-500:]}

@app.delete("/cache/{label}")
def cache_delete(label: str, _: None = Depends(_auth)):
    if label in _cache:
        del _cache[label]
        return {"ok": True, "removed": label, "cache_size": len(_cache)}
    raise HTTPException(status_code=404, detail=f"label {label!r} not in cache")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
