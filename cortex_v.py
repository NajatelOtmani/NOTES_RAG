"""
Notes Assistant — a local RAG app (Streamlit + ChromaDB + Ollama)

Run:   streamlit run cortex_v.py
Needs: .streamlit/config.toml next to this file (light theme, dark sidebar, hides the Deploy button)
Index: python build_index.py   (builds notes_rag_200 / _500 / _1000; without them the app uses "notes_rag")
Eval:  optional eval_set.json -> [{"question": "...", "expected_source": "file.pdf"}, ...]
"""

import os

# Silence terminal noise — must be set before the imports below
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import html
import json
import re
import threading
import time
from collections import Counter

import chromadb
import numpy as np
import ollama
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer

# ───────────────────────── Configuration ─────────────────────────
APP_NAME = "Notes Assistant"
DB_DIR = "./chroma_db"
OLLAMA_MODEL = "llama3.2"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "notes_rag"
CHUNK_COLLECTIONS = {200: "notes_rag_200", 500: "notes_rag_500", 1000: "notes_rag_1000"}
EVAL_FILE = "eval_set.json"
GUARDRAIL_PHRASE = "I don't know based on the provided notes."
SIM_FLOOR = 0.20  # below this best-cosine, the LLM is not called at all (tune it)
GREETINGS = {"hello", "hi", "hey", "hola", "salut", "bonjour", "salam", "good morning", "good evening"}
THANKS = {"thanks", "thank you", "thx", "merci", "ok thanks", "great thanks"}
SUGGESTIONS = [  # edit freely: shown on the empty screen
    "How do I handle missing data and outliers?",
    "Explain the bias-variance tradeoff",
    "What metrics evaluate a classifier?",
    "How should I prepare for an ML interview?",
]
ALIASES = {  # abbreviations expanded for the search only (edit freely)
    "dl": "deep learning", "ml": "machine learning", "nlp": "natural language processing",
    "cnn": "convolutional neural network", "rnn": "recurrent neural network", "llm": "large language model",
    "nn": "neural network", "rl": "reinforcement learning", "cv": "computer vision",
}
FOLLOW_UP = re.compile(r"\b(it|this|that|them|these|those|they|more|detail|details|detailed|elaborate|explain|again|"
                       r"example|examples|why)\b", re.I)
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}

LOGO_SVG = (
    '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5V5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/>'
    '<path d="M8 7h7M8 11h5"/></svg>'
)

st.set_page_config(page_title=APP_NAME, page_icon="📚", layout="wide")

# ───────────────────────── Styling ─────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--ink:#0f172a;--muted:#64748b;--line:#e5e7eb;--soft:#f8fafc;--blue:#2563eb;--blue-d:#1d4ed8;--side:#0b1220}
html, body, .stApp, [data-testid="stMain"]{font-family:'Inter',sans-serif}
.stApp{background:#fff;color:var(--ink)}
#MainMenu, footer, [data-testid="stAppDeployButton"], .stDeployButton{display:none!important}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stStatusWidget"]{display:none!important}
.block-container{max-width:820px;padding-top:2.2rem;padding-bottom:7rem}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1220,#080d18);border-right:1px solid #1a2438}
section[data-testid="stSidebar"] *{color:#e2e8f0}
.brand{display:flex;align-items:center;gap:.65rem;font-weight:700;font-size:1.05rem;margin:.3rem 0 1.2rem;letter-spacing:-.01em}
.mark{width:32px;height:32px;border-radius:9px;display:grid;place-items:center;
      background:linear-gradient(135deg,#2563eb,#7c3aed);box-shadow:0 6px 16px -6px rgba(37,99,235,.7)}
.side-label{display:flex;justify-content:space-between;font-size:.68rem;font-weight:600;letter-spacing:.12em;
            text-transform:uppercase;color:#64748b!important;margin:1.4rem .2rem .5rem}
.doc{display:flex;align-items:center;gap:.65rem;padding:.5rem .55rem;border-radius:10px;font-size:.84rem;transition:background .15s}
.doc:hover{background:#111b30}
.pdf{flex-shrink:0;font-size:.55rem;font-weight:700;letter-spacing:.04em;padding:.22rem .32rem;border-radius:6px;
     background:rgba(248,113,113,.14);color:#fca5a5!important}
.doc-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.doc-n{flex-shrink:0;font-size:.7rem;color:#94a3b8!important;background:#111b30;border-radius:999px;padding:.1rem .5rem}
.status{display:flex;align-items:center;gap:.6rem;margin-top:1.6rem;padding:.7rem .8rem;border:1px solid #1a2438;
        border-radius:12px;background:#0d1627;font-size:.78rem}
.status b{display:block;font-weight:600;font-size:.8rem}
.status small{color:#94a3b8!important}
.dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 4px rgba(34,197,94,.15);flex-shrink:0}
.dot.off{background:#ef4444;box-shadow:0 0 0 4px rgba(239,68,68,.15)}
section[data-testid="stSidebar"] .stButton>button{width:100%;border-radius:12px}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:var(--blue);border:0;font-weight:600;
      padding:.6rem 1rem;box-shadow:0 8px 20px -10px rgba(37,99,235,.9)}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{background:var(--blue-d)}
section[data-testid="stSidebar"] [data-testid="stExpander"]{border:1px solid #1a2438;border-radius:12px;background:transparent}

/* ── Empty state ── */
.hero{text-align:center;margin:2.2rem 0 1.8rem}
.logo{width:60px;height:60px;border-radius:18px;margin:0 auto 1.1rem;display:grid;place-items:center;
      background:linear-gradient(135deg,#2563eb,#7c3aed);box-shadow:0 18px 36px -14px rgba(37,99,235,.6)}
.title{font-size:2.5rem;font-weight:800;letter-spacing:-.03em;line-height:1.1;
       background:linear-gradient(90deg,#0f172a 30%,#2563eb);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--muted);margin-top:.6rem;font-size:1rem}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin:0 0 2rem}
.stat{border:1px solid var(--line);border-radius:14px;padding:.9rem 1rem;background:var(--soft)}
.stat .n{font-size:1.5rem;font-weight:700;letter-spacing:-.02em}
.stat .l{font-size:.78rem;color:var(--muted);margin-top:.1rem}
.try{font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:0 0 .7rem}
[data-testid="stMain"] .stButton>button{width:100%;text-align:left;justify-content:flex-start;white-space:normal;height:auto;
      background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:14px;padding:1rem 1.1rem;font-weight:500;
      box-shadow:0 1px 2px rgba(15,23,42,.04);transition:all .15s}
[data-testid="stMain"] .stButton>button:hover{border-color:var(--blue);color:var(--blue);background:#fff;
      transform:translateY(-1px);box-shadow:0 10px 24px -14px rgba(37,99,235,.55)}

/* ── Chat ── */
div[data-testid="stChatMessage"]{background:transparent;padding:.5rem 0;gap:.9rem}
div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li{line-height:1.7;font-size:.97rem}
[data-testid="stChatMessageAvatarAssistant"]{background:linear-gradient(135deg,#2563eb,#7c3aed)!important;color:#fff!important;
      border-radius:10px!important}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){flex-direction:row-reverse;margin-left:auto;
      width:fit-content;max-width:80%;background:var(--blue);border-radius:18px 18px 4px 18px;padding:.7rem 1.1rem;
      box-shadow:0 10px 24px -14px rgba(37,99,235,.8)}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p{color:#fff;margin:0}
[data-testid="stChatMessageAvatarUser"]{display:none!important}
[data-testid="stBottom"]>div{background:linear-gradient(180deg,rgba(255,255,255,0),#fff 35%)}
[data-testid="stChatInput"]{border-radius:18px;background:#fff;box-shadow:0 14px 34px -16px rgba(15,23,42,.28)}

/* ── Sources ── */
.src-label{font-size:.68rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:.9rem 0 .35rem}
.meta{font-size:.74rem;color:var(--muted);text-align:right;padding-top:.35rem;white-space:nowrap}
[data-testid="stPopover"] button{border-radius:999px;font-size:.76rem;padding:.12rem .75rem;min-height:0;background:var(--soft);
      border:1px solid var(--line);color:#334155;font-weight:500}
[data-testid="stPopover"] button:hover{border-color:var(--blue);color:var(--blue);background:#fff}
.notice{display:inline-block;margin:.2rem 0 .3rem;padding:.4rem .85rem;border-radius:10px;font-size:.82rem;
        color:#92400e;background:#fffbeb;border:1px solid #fde68a}
</style>
""",
    unsafe_allow_html=True,
)


# ───────────────────────── Backend ─────────────────────────
@st.cache_resource
def load_embedder():
    try:
        return SentenceTransformer(EMBED_MODEL, local_files_only=True)  # no HF Hub call
    except Exception:
        return SentenceTransformer(EMBED_MODEL)  # first run: download


@st.cache_resource
def load_client():
    return chromadb.PersistentClient(path=DB_DIR)


def collection_names():
    return [c if isinstance(c, str) else c.name for c in load_client().list_collections()]


def get_collection(name):
    return load_client().get_collection(name=name)


@st.cache_data(show_spinner=False)
def vault_stats(coll_name):
    data = get_collection(coll_name).get(include=["metadatas"])
    return dict(Counter(m.get("source", "unknown") for m in data["metadatas"]))


@st.cache_data(show_spinner=False)
def sample_vectors(coll_name, n=250):
    data = get_collection(coll_name).get(limit=n, include=["embeddings"])
    return np.array(data["embeddings"])


@st.cache_data(ttl=10, show_spinner=False)
def ollama_status():
    try:
        models = ollama.list()
        items = getattr(models, "models", None) or models.get("models", [])
        names = [getattr(m, "model", None) or m.get("name", "") for m in items]
        return any(n.startswith(OLLAMA_MODEL) for n in names)
    except Exception:
        return False


@st.cache_resource
def warm_up_llm():
    """Load the model into RAM in the background so the first question isn't slow."""
    def _go():
        try:
            ollama.generate(model=OLLAMA_MODEL, prompt="", keep_alive="30m")
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()
    return True


def retrieve(query, coll_name, k):
    model = load_embedder()
    col = get_collection(coll_name)
    qv = model.encode([query], normalize_embeddings=True)[0]
    res = col.query(query_embeddings=[qv.tolist()], n_results=k,
                    include=["documents", "metadatas", "embeddings"])
    emb = np.array(res["embeddings"][0])
    sims = emb @ qv / (np.linalg.norm(emb, axis=1) * np.linalg.norm(qv) + 1e-9)  # true cosine
    return {"docs": res["documents"][0], "metas": res["metadatas"][0], "sims": sims.tolist(),
            "qv": qv, "emb": emb, "collection": coll_name}


def build_prompt(query, docs, metas):
    context = "".join(f"--- Source [{m.get('source', '?')}]:\n{d}\n\n" for d, m in zip(docs, metas))
    return f"""You are a helpful assistant. Answer the question using ONLY the provided context below.
If the answer cannot be found in the context, state clearly: "{GUARDRAIL_PHRASE}" Do not make things up.
Be concise (about 6 sentences at most) unless the question asks for details.

Context:
{context}

Question: {query}
Answer:"""


def search_query(query, messages):
    """Return (text used for the vector search, question given to the LLM).
    Expands abbreviations (DL -> deep learning) and, for short follow-ups like
    "explain it in details", reuses the previous user question so the search has a topic."""
    def sub(m):
        full = ALIASES.get(m.group(0).lower())
        return f"{m.group(0)} ({full})" if full else m.group(0)

    search_q = re.sub(r"\b[A-Za-z]{2,3}\b", sub, query)
    previous = [m["content"] for m in messages[:-1] if m["role"] == "user"]
    if previous and len(query.split()) <= 8 and FOLLOW_UP.search(query):
        return f"{previous[-1]} {search_q}", f'{query} (follow-up to the earlier question: "{previous[-1]}")'
    return search_q, query


def token_stream(prompt):
    for chunk in ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], stream=True,
                             keep_alive="30m", options={"temperature": 0.1, "num_ctx": 2048, "num_predict": 400}):
        yield chunk["message"]["content"]


def small_talk_reply(q):
    t = re.sub(r"[^\w\s']", "", q.lower()).strip()
    if t in GREETINGS:
        return "Hello! Ask me anything about your notes."
    if t in THANKS:
        return "You're welcome! Anything else about your notes?"
    return None


def run_eval(coll_name, k):
    if not os.path.exists(EVAL_FILE):
        return None
    with open(EVAL_FILE, encoding="utf-8") as f:
        cases = json.load(f)
    hits = sum(
        any(m.get("source") == c["expected_source"] for m in retrieve(c["question"], coll_name, k)["metas"])
        for c in cases
    )
    return round(100 * hits / max(len(cases), 1)), len(cases), k


# ───────────────────────── Detail graph ─────────────────────────
def xray_figure(r):
    k = len(r["docs"])
    pts = np.vstack([r["qv"][None, :], r["emb"], sample_vectors(r["collection"])])
    _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    xy = (pts - pts.mean(axis=0)) @ vt[:2].T
    q, nb, other = xy[0], xy[1 : 1 + k], xy[1 + k :]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=other[:, 0], y=other[:, 1], mode="markers", name="your notes",
                             marker=dict(size=4, color="rgba(100,116,139,.35)"), hoverinfo="skip"))
    for i in range(k):
        fig.add_trace(go.Scatter(x=[q[0], nb[i, 0]], y=[q[1], nb[i, 1]], mode="lines", showlegend=False,
                                 line=dict(color="rgba(37,99,235,.5)", width=1 + 3 * max(r["sims"][i], 0)),
                                 hoverinfo="skip"))
        fig.add_annotation(x=(q[0] + nb[i, 0]) / 2, y=(q[1] + nb[i, 1]) / 2, text=f"{r['sims'][i]:.2f}",
                           showarrow=False, font=dict(color="#2563eb", size=11), bgcolor="rgba(255,255,255,.9)")
    fig.add_trace(go.Scatter(
        x=nb[:, 0], y=nb[:, 1], mode="markers+text", name="matches", text=[f"[{i + 1}]" for i in range(k)],
        textposition="top center",
        marker=dict(size=[12 + 14 * max(s, 0) for s in r["sims"]], color="#2563eb", line=dict(width=1, color="#fff")),
        hovertext=[f"{m.get('source', '?')} · similarity {s:.2f}" for m, s in zip(r["metas"], r["sims"])],
        hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[q[0]], y=[q[1]], mode="markers+text", name="your question", text=["question"],
                             textposition="bottom center",
                             marker=dict(size=15, symbol="diamond", color="#0f172a")))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#0f172a"),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.05))
    return fig


# ───────────────────────── Rendering helpers ─────────────────────────
def short(s, n=18):
    s = os.path.splitext(s)[0]
    return s if len(s) <= n else s[: n - 1] + "…"


def render_extras(msg, idx):
    r = msg.get("retrieval")
    if r is None:  # greeting / small talk
        return
    if msg["guardrail"]:
        st.markdown('<div class="notice">⚠ Not found in your notes — nothing was guessed.</div>',
                    unsafe_allow_html=True)
    n = len(r["docs"])
    st.markdown('<div class="src-label">Sources</div>', unsafe_allow_html=True)
    cols = st.columns([1.7] * n + [1.1, 1.6])
    for i, (doc, meta, sim) in enumerate(zip(r["docs"], r["metas"], r["sims"])):
        label = f":material/description: {short(meta.get('source', '?'))} · {max(sim, 0) * 100:.0f}%"
        with cols[i].popover(label):
            st.markdown(f"**{meta.get('source', '?')}** · similarity {sim:.2f}")
            st.text(doc)
    with cols[n].popover(":material/scatter_plot: Details"):
        if "fig" not in msg:
            msg["fig"] = xray_figure(r)  # computed once
        st.plotly_chart(msg["fig"], key=f"xray_{idx}")
        st.caption("How your question relates to the matching passages. Display is a 2D approximation; "
                   "the numbers are true similarity scores.")
    if msg.get("elapsed"):
        cols[-1].markdown(f'<div class="meta">{n} passages · {msg["elapsed"]:.1f}s</div>', unsafe_allow_html=True)


def render_message(msg, idx):
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg, idx)


# ───────────────────────── State & startup ─────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("hit_rate", None)
load_embedder()  # loaded once at server start, so the first question is fast

existing = collection_names()
chunk_options = [cs for cs, n in CHUNK_COLLECTIONS.items() if n in existing]

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown(f'<div class="brand"><div class="mark">{LOGO_SVG.format(s=17)}</div>{APP_NAME}</div>',
                unsafe_allow_html=True)
    if st.button(":material/add:  New chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Settings"):
        if chunk_options:
            chunk_size = st.radio("Chunk size", chunk_options, horizontal=True,
                                  index=min(1, len(chunk_options) - 1))
            coll_name = CHUNK_COLLECTIONS[chunk_size]
        else:
            coll_name = DEFAULT_COLLECTION
        top_k = st.radio("Passages per answer", [2, 3, 5], horizontal=True, index=1)
        if st.button("Test retrieval accuracy"):
            out = run_eval(coll_name, top_k)
            st.session_state.hit_rate = out
            if out is None:
                st.caption(f"Add {EVAL_FILE} next to the app to enable this.")
        if st.session_state.hit_rate:
            acc, n_q, k_used = st.session_state.hit_rate
            st.caption(f"Accuracy: {acc}% (top-{k_used}, {n_q} questions)")

    try:
        get_collection(coll_name)
    except Exception:
        st.error(f"Collection '{coll_name}' not found. Run your indexing script first.")
        st.stop()

    stats = vault_stats(coll_name)
    total_chunks = sum(stats.values())
    st.markdown(f'<div class="side-label"><span>Documents</span><span>{len(stats)}</span></div>',
                unsafe_allow_html=True)
    for src, n in sorted(stats.items(), key=lambda kv: kv[0].lower()):
        ext = (os.path.splitext(src)[1][1:] or "file").upper()[:4]
        st.markdown(
            f'<div class="doc" title="{html.escape(src)}"><span class="pdf">{ext}</span>'
            f'<span class="doc-name">{html.escape(os.path.splitext(src)[0])}</span>'
            f'<span class="doc-n">{n}</span></div>',
            unsafe_allow_html=True,
        )

    online = ollama_status()
    if online:
        warm_up_llm()
    st.markdown(
        f'<div class="status"><span class="dot{"" if online else " off"}"></span>'
        f'<div><b>{OLLAMA_MODEL}</b><small>{"Running locally" if online else "Ollama is offline"}</small></div></div>',
        unsafe_allow_html=True,
    )

# ───────────────────────── Main ─────────────────────────
query = st.chat_input("Ask a question about your notes…")
if "pending" in st.session_state:  # a suggestion was clicked
    query = st.session_state.pop("pending")
if query:
    st.session_state.messages.append({"role": "user", "content": query})

if not st.session_state.messages:
    st.markdown(
        f'<div class="hero"><div class="logo">{LOGO_SVG.format(s=28)}</div>'
        f'<div class="title">{APP_NAME}</div>'
        f'<div class="sub">Answers come only from your own documents.</div></div>'
        f'<div class="stats">'
        f'<div class="stat"><div class="n">{len(stats)}</div><div class="l">Documents</div></div>'
        f'<div class="stat"><div class="n">{total_chunks:,}</div><div class="l">Passages indexed</div></div>'
        f'<div class="stat"><div class="n">100% local</div><div class="l">{OLLAMA_MODEL} · nothing leaves your machine</div></div>'
        f'</div><div class="try">Suggested questions</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if (c1 if i % 2 == 0 else c2).button(q, key=f"sugg_{i}"):
            st.session_state.pending = q
            st.rerun()
else:
    for idx, msg in enumerate(st.session_state.messages):
        render_message(msg, idx)

if query:
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        reply = small_talk_reply(query)
        if reply:
            st.markdown(reply)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "retrieval": None, "guardrail": False})
        else:
            t0 = time.perf_counter()
            search_q, llm_q = search_query(query, st.session_state.messages)
            r = retrieve(search_q, coll_name, top_k)
            if max(r["sims"]) < SIM_FLOOR:  # nothing relevant: don't let the LLM improvise
                answer = GUARDRAIL_PHRASE
                st.markdown(answer)
            else:
                try:
                    answer = st.write_stream(token_stream(build_prompt(llm_q, r["docs"], r["metas"])))
                except Exception as e:
                    st.error(f"Ollama error: {e}")
                    st.stop()
            guardrail = GUARDRAIL_PHRASE.lower().rstrip(".") in answer.lower()
            msg = {"role": "assistant", "content": answer, "retrieval": r, "guardrail": guardrail,
                   "elapsed": time.perf_counter() - t0}
            st.session_state.messages.append(msg)
            render_extras(msg, len(st.session_state.messages) - 1)