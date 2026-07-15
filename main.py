"""
chainstate-priors — nightly corpus ingester for CHAINSTATE v0.7.0

Pulls fresh content from four source families and writes empirical
symbol distributions + semantic embeddings into Cloudflare KV, where
the CHAINSTATE worker can consult them as priors during query handling.

Sources:
  1. Wikipedia (English)  — /api/rest_v1/page/summary of a curated topic set
  2. arXiv                — recent abstracts from cs.LG, cs.AI, math.CO, quant-ph
  3. Ecosystem HF Spaces  — the eight+ Spaces that link out from NWO Agentic:
     nwo-agentic, nwo-chainstate, ornith-chainstate, chainstate-chat,
     nwo-neuro, nwo-genetic, nwo-blackbox, nwo-cardiac, nwo-geohack,
     nwo-mr, ornith-mr, ornith, nwo-anon, publicae, metastate
  4. ResearchGate         — publication summaries via public search
     (Casimir-Sonoluminescence, CHAINSTATE, CHAINSTATE CODE, ASI-Evolve,
      NWO-ASM, Distributed LM Agent, and any others you add)

For each ingested item we compute:
  • 384-dim embedding via chainstate-encoder
  • per-subspace symbol distribution (same routing math as the worker)
  • a short summary (first ~500 chars, plaintext)
  • source URL, title, timestamp

And write to Cloudflare KV under two prefixes:
  prior:{source}:{slug}  →  { summary, subspace_dist, ts, url, title }
  vec:{source}:{slug}    →  { vec: [384 floats], ts }

The CHAINSTATE worker has new endpoints /priors/query and /priors/list
that read from these prefixes (see worker patch in this delivery).

Runs as a Render Background Worker with a schedule of `0 3 * * *`
(daily at 03:00 UTC — off-peak, gentle on arXiv & Wikipedia mirrors).
Can also be triggered manually via GET /run (auth-gated).

Env vars required:
  ENCODER_URL          → https://chainstate-encoder.onrender.com
  CLOUDFLARE_ACCOUNT_ID → your Cloudflare account ID
  CLOUDFLARE_KV_NAMESPACE_ID → the CHAINSTATE_CACHE KV namespace ID
  CLOUDFLARE_API_TOKEN → API token with KV write permission
  ENCODER_API_KEY      → optional; forwarded to encoder if set
  RUN_TOKEN            → bearer token required to trigger /run manually

Owner: Ciprian Florin Pater
"""
import os
import re
import json
import time
import asyncio
import hashlib
from typing import List, Optional
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse

SERVICE_VER  = "0.7.0-priors-2026-07-15"
ENCODER_URL  = os.getenv("ENCODER_URL",  "https://chainstate-encoder.onrender.com").rstrip("/")
CF_ACCOUNT   = os.getenv("CLOUDFLARE_ACCOUNT_ID",  "")
CF_KV_NS     = os.getenv("CLOUDFLARE_KV_NAMESPACE_ID",  "")
CF_TOKEN     = os.getenv("CLOUDFLARE_API_TOKEN", "")
ENC_API_KEY  = os.getenv("ENCODER_API_KEY", "").strip()
RUN_TOKEN    = os.getenv("RUN_TOKEN", "").strip()

# ─── Subspace routing (identical to worker's tables) ────────────────────
SUBSPACE_SAMPLES = {
    "math": list("∫∂∇∆∑∏∈∉∪∩∀∃⊕⊗∞∝≈≠≤≥≡√∛⌊⌋"),
    "sci":  list("ℏℵℂℕℚℝℤℙℍ⚗⚛🧬🧪🦠🔬🔭🔮☢☣⚡🌡🩺⚕🧲🌊"),
    "lang": list("ΑΒΓΔΕαβγδАБВГ一二三道心学智ابتثאבגअआक가나다라마한국"),
    "occ":  list("☉☽☿♀♁♂♃♄☤☥☦☧☪☮☯✝✠♈♉♊♋🜀🜁🜂🜃🜄🜅🜆"),
    "emo":  list("😀😎🤔🧠👽🤖🐉🦠🌍🌐⛓🔗💎🎯🚀✨🔥💧🌟⚡"),
    "ctrl": list("⇒⇐⇑⇓⇔↺↻⟳⟲⇄⇆⇋⇌→←↑↓↔↕⟶⟵⟷⟸⟹⟺⤴⤵"),
}
_LATIN_RE = re.compile(r"[A-Za-z]")
_NUM_RE   = re.compile(r"[0-9]")

def compute_subspace_dist(text: str) -> dict:
    counts = {k: 0 for k in SUBSPACE_SAMPLES}
    for c in text:
        for k, arr in SUBSPACE_SAMPLES.items():
            if c in arr:
                counts[k] += 1
        if _LATIN_RE.match(c): counts["lang"] += 1
        if _NUM_RE.match(c):   counts["math"] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}

def slug(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9-]+", "-", s.strip().lower()).strip("-")
    if len(s) > maxlen:
        s = s[:maxlen].rstrip("-")
    if not s:
        s = "unknown-" + hashlib.sha256(s.encode() or b"empty").hexdigest()[:8]
    return s

# ─── Cloudflare KV client ────────────────────────────────────────────────
class CFKV:
    def __init__(self, account: str, ns: str, token: str):
        if not (account and ns and token):
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID + KV_NAMESPACE_ID + API_TOKEN required")
        self.base = (
            f"https://api.cloudflare.com/client/v4/accounts/{account}"
            f"/storage/kv/namespaces/{ns}"
        )
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def put(self, client: httpx.AsyncClient, key: str, value: str, ttl_seconds: Optional[int] = None):
        url = f"{self.base}/values/{key}"
        params = {}
        if ttl_seconds is not None:
            params["expiration_ttl"] = str(ttl_seconds)
        r = await client.put(
            url,
            headers={"Authorization": self.headers["Authorization"], "Content-Type": "text/plain"},
            content=value,
            params=params,
        )
        return r.status_code, r.text[:200]

    async def bulk_put(self, client: httpx.AsyncClient, items: list, ttl_seconds: Optional[int] = None):
        # items: [{ "key": str, "value": str }]
        url = f"{self.base}/bulk"
        payload = []
        for item in items:
            entry = {"key": item["key"], "value": item["value"]}
            if ttl_seconds is not None:
                entry["expiration_ttl"] = ttl_seconds
            payload.append(entry)
        r = await client.put(url, headers=self.headers, content=json.dumps(payload))
        return r.status_code, r.text[:200]

# ─── Encoder client ─────────────────────────────────────────────────────
async def embed(client: httpx.AsyncClient, text: str) -> Optional[list]:
    headers = {"Content-Type": "application/json"}
    if ENC_API_KEY:
        headers["Authorization"] = f"Bearer {ENC_API_KEY}"
    try:
        r = await client.post(
            f"{ENCODER_URL}/embed",
            json={"text": text[:8000], "normalize": True},
            headers=headers,
            timeout=30.0,
        )
        if r.status_code == 200:
            return r.json().get("vector")
    except Exception:
        return None
    return None

# ─── Source: Wikipedia REST v1 summary ──────────────────────────────────
WIKIPEDIA_TOPICS = [
    # Symbol / semiotics / language
    "Symbol", "Semiotics", "Sign_(semiotics)", "Umberto_Eco", "Charles_Sanders_Peirce",
    "Linguistics", "Cognitive_linguistics",
    # Math foundations of what CHAINSTATE routes as math subspace
    "Set_theory", "Category_theory", "Homotopy_type_theory", "Godel's_incompleteness_theorems",
    "Symbolic_computation", "Algorithmic_information_theory", "Kolmogorov_complexity",
    # Science subspace
    "Casimir_effect", "Sonoluminescence", "Zero-point_energy", "Quantum_field_theory",
    "Bell's_theorem", "Bayesian_inference", "Free_energy_principle",
    # Consensus / distributed systems
    "Byzantine_fault_tolerance", "Distributed_consensus", "Blockchain", "Merkle_tree",
    "Directed_acyclic_graph", "Consensus_(computer_science)",
    # AI / cognition / AGI
    "Artificial_general_intelligence", "Reinforcement_learning_from_human_feedback",
    "Transformer_(deep_learning_architecture)", "Attention_(machine_learning)",
    "Mixture_of_experts", "Sparse_representation", "Symbolic_artificial_intelligence",
    "Neurosymbolic_AI", "Genetic_programming", "Program_synthesis",
    # Modal logic
    "Modal_logic", "Kripke_semantics", "Deontic_logic", "Epistemic_logic",
    "Doxastic_logic", "Dynamic_logic_(modal_logic)",
    # Occult / esoteric (occ subspace)
    "Alchemy", "Hermeticism", "Kabbalah", "I_Ching", "Astrology", "Sacred_geometry",
    # Cryptography
    "SHA-3", "Elliptic-curve_cryptography", "Zero-knowledge_proof",
    # Sovereignty / governance
    "Digital_nation_state", "Network_state", "Sovereignty", "Autonomous_organization",
]

async def ingest_wikipedia(client: httpx.AsyncClient, kv: CFKV) -> int:
    count = 0
    for topic in WIKIPEDIA_TOPICS:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            r = await client.get(url, headers={"User-Agent": "chainstate-priors/0.7"}, timeout=15.0)
            if r.status_code != 200:
                continue
            data = r.json()
            title   = data.get("title") or topic.replace("_", " ")
            summary = (data.get("extract") or "").strip()
            page_url = (data.get("content_urls", {}).get("desktop", {}).get("page")
                        or f"https://en.wikipedia.org/wiki/{topic}")
            if len(summary) < 60:
                continue
            vec = await embed(client, f"{title}. {summary}")
            dist = compute_subspace_dist(summary)
            key_id = slug(topic)
            record = {
                "source": "wikipedia",
                "title": title,
                "summary": summary[:1500],
                "url": page_url,
                "subspace_dist": dist,
                "ts": datetime.now(timezone.utc).isoformat(),
                "ingester_version": SERVICE_VER,
            }
            await kv.put(client, f"prior:wikipedia:{key_id}", json.dumps(record), ttl_seconds=7*86400)
            if vec:
                await kv.put(client, f"vec:wikipedia:{key_id}",
                             json.dumps({"vec": vec, "ts": record["ts"]}), ttl_seconds=7*86400)
            count += 1
            await asyncio.sleep(0.2)  # gentle
        except Exception as e:
            print(f"[wikipedia] {topic}: {e}", flush=True)
    return count

# ─── Source: arXiv recent listings ──────────────────────────────────────
ARXIV_CATEGORIES = ["cs.LG", "cs.AI", "cs.CL", "cs.DC", "cs.CR", "math.CO", "math.LO", "quant-ph"]

async def ingest_arxiv(client: httpx.AsyncClient, kv: CFKV, per_cat: int = 5) -> int:
    count = 0
    for cat in ARXIV_CATEGORIES:
        try:
            url = ("http://export.arxiv.org/api/query?"
                   f"search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&max_results={per_cat}")
            r = await client.get(url, timeout=20.0)
            if r.status_code != 200:
                continue
            entries = _parse_arxiv_atom(r.text)
            for e in entries:
                title = e["title"]
                summary = e["summary"]
                arxiv_id = e["id"].split("/abs/")[-1].split("v")[0]
                vec = await embed(client, f"{title}. {summary}")
                dist = compute_subspace_dist(summary)
                record = {
                    "source": "arxiv",
                    "category": cat,
                    "title": title,
                    "summary": summary[:1500],
                    "url": e["id"],
                    "subspace_dist": dist,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "ingester_version": SERVICE_VER,
                }
                key_id = slug(f"{cat}-{arxiv_id}")
                await kv.put(client, f"prior:arxiv:{key_id}", json.dumps(record), ttl_seconds=14*86400)
                if vec:
                    await kv.put(client, f"vec:arxiv:{key_id}",
                                 json.dumps({"vec": vec, "ts": record["ts"]}), ttl_seconds=14*86400)
                count += 1
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"[arxiv] {cat}: {e}", flush=True)
    return count

def _parse_arxiv_atom(xml_text: str) -> list:
    # Minimal Atom parser — avoids feedparser dep
    items = []
    entries = re.split(r"<entry>", xml_text)[1:]
    for entry in entries:
        def grab(tag):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", entry, re.DOTALL)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        title   = grab("title")
        summary = grab("summary")
        id_     = grab("id")
        if title and summary and id_:
            items.append({"title": title, "summary": summary, "id": id_})
    return items

# ─── Source: Ecosystem HF Spaces (linked from nwo-agentic index.html) ──
ECOSYSTEM_HF_SPACES = [
    # Format: (slug, human_title, description)
    ("nwo-agentic",       "NWO Agentic",       "Multi-agent orchestration hub for the NWO ecosystem — links to all Spaces below"),
    ("chainstate",        "CHAINSTATE",        "Symbolic-weight blockchain on Base mainnet 8453 — 65,536-dim symbol space, six subspaces, reputation-weighted Bayesian log-pooling"),
    ("ornith-chainstate", "Ornith × CHAINSTATE","Ornith-1.0 (9B/35B/397B MoE) coding agent × CHAINSTATE — builder, terminal, agent, simulation, AGI dashboard with ASI-Evolve loop"),
    ("chainstate-chat",   "CHAINSTATE Chat",   "Two-pane chat interface to the symbolic-weight blockchain — polished answer left, live receipt trace right"),
    ("metastate",         "METASTATE",         "Free-energy anomaly substrate — Odrzywołek EML symbolic regression + TimesFM 2.5 temporal prior"),
    ("nwo-neuro",         "NWO NEURO",         "Brain-computer interface substrate — Mental State Signatures (focus, valence, arousal, cognitive_load, intent) from EEG"),
    ("nwo-genetic",       "NWO GENETIC",       "Biological compiler OS — 12-layer architecture, 6 write backends, post-quantum crypto, on-chain USDC settlement"),
    ("nwo-blackbox",      "NWO BLACKBOX",      "Off-grid mission control platform — sovereignty in extremis, survival protocols"),
    ("nwo-cardiac",       "NWO Cardiac",       "ECG-bound soul-bound identity SDK — cardiac-hash identity anchoring on Base 8453"),
    ("nwo-geohack",       "NWO GEOHACK",       "Open-world geo-hacking RPG and sovereign-internet client — 16-tier rank ladder, $STATE tokenomics"),
    ("nwo-mr",            "NWO MR",            "Mixed reality substrate — Gaussian splat worlds, 3D mesh generation, NFT minting on Base 8453"),
    ("nwo-anon",          "NWO ANON",          "Anonymous transaction layer for the NWO ecosystem"),
    ("publicae",          "Imperium Romanum Digital Nation State", "Publicae — the sovereign nation-state framework built on Base mainnet 8453, ministry stack, DAO governance"),
    ("nwo-capital",       "NWO Capital",       "Portfolio SPA + backend — deal flow, robotics investments, RWA marketplace"),
    ("nwo-rwa",           "NWO RWA",           "Real-world asset marketplace — compliance-gated tokenization on Base 8453"),
    ("nwo-zeropoint",     "NWO ZeroPoint",     "Casimir-Sonoluminescence zero-point energy device Web3 stack — token, fund, NFT, revenue, bid market"),
    ("nwo-coanda",        "NWO COANDA",        "VTOL flying car presale application — smart contract on Base 8453"),
    ("nwo-ubi",           "NWO UBI",           "Universal Basic Income via $STATE — EIP-2612 permit-based claim flow"),
    ("nwo-asi",           "NWO ASI",           "Tokenized vacuum-field thruster investment pool — Uniswap V2 TWAP, ERC-1155 hybrid tokens"),
]
HF_USER = "CPater"

async def ingest_ecosystem_spaces(client: httpx.AsyncClient, kv: CFKV) -> int:
    count = 0
    for hf_slug, title, description in ECOSYSTEM_HF_SPACES:
        # Try to fetch the space's own metadata via HF API
        summary_text = f"{title}. {description}"
        space_url = f"https://huggingface.co/spaces/{HF_USER}/{hf_slug}"
        try:
            r = await client.get(
                f"https://huggingface.co/api/spaces/{HF_USER}/{hf_slug}",
                timeout=10.0,
            )
            if r.status_code == 200:
                meta = r.json()
                card = meta.get("cardData") or {}
                sd = card.get("short_description") or ""
                if sd:
                    summary_text += f" · {sd}"
        except Exception:
            pass
        vec = await embed(client, summary_text)
        dist = compute_subspace_dist(summary_text)
        record = {
            "source": "ecosystem_hf_space",
            "title": title,
            "summary": summary_text[:1500],
            "url": space_url,
            "subspace_dist": dist,
            "ts": datetime.now(timezone.utc).isoformat(),
            "ingester_version": SERVICE_VER,
        }
        key_id = slug(hf_slug)
        # Ecosystem priors are relatively stable — TTL 30 days (or set no TTL)
        await kv.put(client, f"prior:ecosystem:{key_id}", json.dumps(record), ttl_seconds=30*86400)
        if vec:
            await kv.put(client, f"vec:ecosystem:{key_id}",
                         json.dumps({"vec": vec, "ts": record["ts"]}), ttl_seconds=30*86400)
        count += 1
        await asyncio.sleep(0.1)
    return count

# ─── Source: ResearchGate publications (Ciprian Florin Pater's corpus) ─
# ResearchGate doesn't have a stable public API, but the publication
# summary pages are cacheable HTML. We seed with your DOIs / RG IDs
# and fetch the abstract from the page. Structured summaries below.
RESEARCHGATE_PUBLICATIONS = [
    {
        "rg_id": "407489249",
        "title": "Casimir-Sonoluminescence Coupling · Physics Essays",
        "abstract": ("Zero-point vacuum fluctuations coupled to sonoluminescent bubble collapse. "
                     "Proposes a mechanism by which nonlinear acoustic focusing modulates the local "
                     "Casimir energy density, producing measurable photon emission. Peer-reviewed in "
                     "Physics Essays. Foundation for NWO ZeroPoint device design."),
        "url": "https://www.researchgate.net/publication/407489249",
    },
    {
        "rg_id": "407444375",
        "title": "CHAINSTATE Whitepaper v1.0",
        "abstract": ("A symbolic-weight blockchain on Base mainnet 8453. Introduces the 65,536-dim "
                     "symbolic embedding space across six structured subspaces (math, sci, lang, occ, "
                     "emo, ctrl). Consensus via reputation-weighted Bayesian log-pooling. Convergence "
                     "in 3–7 rounds at cosine ≥ 0.95. Settlement in USDC. Cache TTL 5 minutes."),
        "url": "https://www.researchgate.net/publication/407444375",
    },
    {
        "rg_id": "408393584",
        "title": "CHAINSTATE CODE · A Formal Framework for Agentic Coding on a Symbolic-Weight Blockchain",
        "abstract": ("Extends CHAINSTATE with the Ornith-1.0 (9B/35B/397B MoE) coding agent stack, "
                     "the NWO-ASM Process-Matrix IR, and the ASI-Evolve loop for autonomous program "
                     "evolution. 18 opcodes across 5 groups. Bounded evolutionary search with hard "
                     "Deontic veto in the fitness function."),
        "url": "https://www.researchgate.net/publication/408393584",
    },
    {
        "rg_id": "409148376",
        "title": "Verifiable Autonomous Cognition at the Frontier · CHAINSTATE + ASI-Evolve Integration",
        "abstract": ("21-page peer-reviewed publication. Covers the v0.3.0→v0.6.0 arc with math per "
                     "phase (Eqs. 3–7). Introduces the four-dimensional modal receipt (Epistemic, "
                     "Doxastic, Deontic, Dynamic) with truth lattice L={b,M}^4. Fitness function "
                     "S(π) = 100c − 5000g − 2d with hard Deontic veto S = −∞ if V = REFUSED. "
                     "Compares against Claude Opus 4.8, GPT-5.5, Gemini 3.1 Pro, DeepSeek V4-Pro. "
                     "First-to-market analysis and geopolitical scenario grid."),
        "url": "https://www.researchgate.net/publication/409148376",
    },
    {
        "rg_id": "408502100",
        "title": "NWO-ASM · Process-Matrix Intermediate Representation for Distributed Cognition",
        "abstract": ("39-page LaTeX whitepaper on NWO-ASM. Defines the Process-Matrix IR (.pmx), the "
                     "18 opcodes across 5 groups (Data, Compute, Memory, Control, Metadata), and the "
                     "compile-and-dispatch flow to GPU/QPU/NPU substrates via the CHAINSTATE worker. "
                     "Includes formal semantics, cost model, and reference implementation."),
        "url": "https://www.researchgate.net/publication/408502100",
    },
]

async def ingest_researchgate(client: httpx.AsyncClient, kv: CFKV) -> int:
    count = 0
    for pub in RESEARCHGATE_PUBLICATIONS:
        try:
            summary_text = f"{pub['title']}. {pub['abstract']}"
            vec = await embed(client, summary_text)
            dist = compute_subspace_dist(pub["abstract"])
            record = {
                "source": "researchgate",
                "rg_id": pub["rg_id"],
                "title": pub["title"],
                "summary": pub["abstract"][:1500],
                "url": pub["url"],
                "subspace_dist": dist,
                "ts": datetime.now(timezone.utc).isoformat(),
                "author": "Ciprian Florin Pater",
                "ingester_version": SERVICE_VER,
            }
            key_id = slug(pub["rg_id"])
            # Author's own corpus — long TTL (90d)
            await kv.put(client, f"prior:researchgate:{key_id}", json.dumps(record), ttl_seconds=90*86400)
            if vec:
                await kv.put(client, f"vec:researchgate:{key_id}",
                             json.dumps({"vec": vec, "ts": record["ts"]}), ttl_seconds=90*86400)
            count += 1
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[researchgate] {pub['rg_id']}: {e}", flush=True)
    return count

# ─── Orchestration ──────────────────────────────────────────────────────
async def run_ingest_all() -> dict:
    if not (CF_ACCOUNT and CF_KV_NS and CF_TOKEN):
        return {"error": "Cloudflare KV credentials not configured"}
    kv = CFKV(CF_ACCOUNT, CF_KV_NS, CF_TOKEN)
    t0 = time.time()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Verify encoder is reachable first — no point ingesting if we can't embed
        try:
            h = await client.get(f"{ENCODER_URL}/health", timeout=10.0)
            if h.status_code != 200:
                return {"error": f"encoder /health returned {h.status_code}", "encoder": ENCODER_URL}
        except Exception as e:
            return {"error": f"encoder unreachable: {e}", "encoder": ENCODER_URL}

        results = {}
        results["ecosystem"]    = await ingest_ecosystem_spaces(client, kv)
        results["researchgate"] = await ingest_researchgate(client, kv)
        results["wikipedia"]    = await ingest_wikipedia(client, kv)
        results["arxiv"]        = await ingest_arxiv(client, kv)
    return {
        "ok": True,
        "ingested_counts": results,
        "elapsed_s": round(time.time() - t0, 1),
        "encoder": ENCODER_URL,
        "kv_namespace": CF_KV_NS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service_version": SERVICE_VER,
    }

# ─── FastAPI surface ─────────────────────────────────────────────────────
app = FastAPI(title="chainstate-priors", version=SERVICE_VER)

def _run_auth(authorization: Optional[str] = Header(None)) -> None:
    if not RUN_TOKEN:
        raise HTTPException(status_code=503, detail="RUN_TOKEN not configured — cannot trigger manually")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1].strip() != RUN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

@app.get("/")
def welcome():
    return {
        "service": "chainstate-priors",
        "version": SERVICE_VER,
        "purpose": "Nightly corpus ingester — Wikipedia + arXiv + ecosystem HF Spaces + ResearchGate → Cloudflare KV",
        "endpoints": [
            "GET  /              → this page",
            "GET  /health        → readiness + config status",
            "POST /run           → trigger ingest (auth-gated with RUN_TOKEN)",
            "GET  /schedule      → next scheduled run info",
        ],
        "owner": "Ciprian Florin Pater",
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "encoder_configured": bool(ENCODER_URL),
        "cloudflare_configured": bool(CF_ACCOUNT and CF_KV_NS and CF_TOKEN),
        "run_token_set": bool(RUN_TOKEN),
        "sources": ["wikipedia", "arxiv", "ecosystem_hf_space", "researchgate"],
        "source_topic_counts": {
            "wikipedia": len(WIKIPEDIA_TOPICS),
            "arxiv_categories": len(ARXIV_CATEGORIES),
            "ecosystem_hf_spaces": len(ECOSYSTEM_HF_SPACES),
            "researchgate_publications": len(RESEARCHGATE_PUBLICATIONS),
        },
    }

@app.post("/run")
async def run_now(_: None = Depends(_run_auth)):
    result = await run_ingest_all()
    return JSONResponse(result)

@app.get("/schedule")
def schedule():
    return {
        "cron": "0 3 * * *",
        "note": "Configured via Render Cron Job — runs daily at 03:00 UTC",
        "next_run_hint": "check Render dashboard → chainstate-priors → Events",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
