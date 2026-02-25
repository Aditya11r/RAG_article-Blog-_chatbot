import os
import io
import sqlite3
import json
import tempfile
import hashlib
from datetime import datetime

import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if not OPENROUTER_API_KEY:
    st.error("❌ OPENROUTER_API_KEY not found.")
    st.stop()

# ── DB path (sits next to app.py) ────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")


# ─────────────────────────────────────────────────────────────
#  DATABASE LAYER
# ─────────────────────────────────────────────────────────────
def get_db():
    """Return a thread-local SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            email     TEXT    NOT NULL,
            username  TEXT    NOT NULL,
            user_key  TEXT    UNIQUE NOT NULL,   -- sha256(lower(email)+lower(username))
            created_at TEXT   NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            session_tag TEXT    NOT NULL,        -- human-readable label
            source_meta TEXT,                   -- JSON: {icon, name, detail, chunks}
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            role       TEXT NOT NULL,            -- 'user' | 'assistant'
            content    TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)


def _user_key(email: str, username: str) -> str:
    raw = f"{email.strip().lower()}::{username.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_or_create_user(email: str, username: str) -> int:
    key = _user_key(email, username)
    conn = get_db()
    row = conn.execute("SELECT id FROM users WHERE user_key=?", (key,)).fetchone()
    if row:
        return row["id"]
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO users (email, username, user_key, created_at) VALUES (?,?,?,?)",
        (email.strip().lower(), username.strip(), key, now),
    )
    conn.commit()
    return cur.lastrowid


def create_session(user_id: int, source_meta: dict | None = None) -> int:
    now = datetime.utcnow().isoformat()
    tag = f"Session {now[:16].replace('T', ' ')}"
    meta_json = json.dumps(source_meta) if source_meta else None
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO sessions (user_id, session_tag, source_meta, created_at, updated_at) VALUES (?,?,?,?,?)",
        (user_id, tag, meta_json, now, now),
    )
    conn.commit()
    return cur.lastrowid


def update_session_source(session_id: int, source_meta: dict):
    now = datetime.utcnow().isoformat()
    conn = get_db()
    conn.execute(
        "UPDATE sessions SET source_meta=?, updated_at=? WHERE id=?",
        (json.dumps(source_meta), now, session_id),
    )
    conn.commit()


def get_user_sessions(user_id: int) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE user_id=? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(session_id: int) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def save_message(session_id: int, role: str, content: str):
    now = datetime.utcnow().isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
        (session_id, role, content, now),
    )
    # bump updated_at on the session
    conn.execute(
        "UPDATE sessions SET updated_at=? WHERE id=?",
        (now, session_id),
    )
    conn.commit()


def delete_session(session_id: int):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()


# ── Init DB once ─────────────────────────────
init_db()


# ─────────────────────────────────────────────
#  PAGE CONFIG & STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Source RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #0d0f14; color: #e2e8f0; }

[data-testid="stSidebar"] { background: #131720 !important; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.app-header { text-align: center; padding: 2rem 0 1rem 0; }
.app-header h1 {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.app-header p { color: #64748b; font-size: 0.95rem; font-weight: 300; }

.doc-card {
    background: #131720; border: 1px solid #1e2535;
    border-left: 3px solid #38bdf8; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    font-size: 0.85rem; color: #94a3b8;
}
.doc-card strong { color: #e2e8f0; font-size: 0.9rem; display: block; margin-bottom: 4px; }
.doc-card .badge {
    display: inline-block; background: #0f172a; border: 1px solid #334155;
    color: #38bdf8; font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; padding: 2px 8px; border-radius: 20px; margin-top: 4px;
}

.session-card {
    background: #131720; border: 1px solid #1e2535; border-radius: 10px;
    padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    font-size: 0.82rem; color: #94a3b8; cursor: pointer;
    transition: border-color 0.2s;
}
.session-card:hover { border-color: #38bdf8; }
.session-card strong { color: #e2e8f0; font-size: 0.85rem; display: block; }

.chat-user { display: flex; justify-content: flex-end; margin: 0.6rem 0; }
.chat-user .bubble {
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: #fff; border-radius: 18px 18px 4px 18px;
    padding: 0.7rem 1.1rem; max-width: 72%;
    font-size: 0.92rem; line-height: 1.5;
}
.chat-bot { display: flex; justify-content: flex-start; margin: 0.6rem 0; }
.chat-bot .bubble {
    background: #1a1f2e; border: 1px solid #1e2535; color: #e2e8f0;
    border-radius: 18px 18px 18px 4px; padding: 0.7rem 1.1rem;
    max-width: 80%; font-size: 0.92rem; line-height: 1.6;
}

.login-box {
    background: #131720; border: 1px solid #1e2535; border-radius: 14px;
    padding: 2rem 2.5rem; max-width: 420px; margin: 3rem auto;
}
.login-box h2 {
    font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.login-box p { color: #64748b; font-size: 0.85rem; margin-bottom: 1.5rem; }

.stTextInput > div > div > input {
    background: #131720 !important; border: 1px solid #1e2535 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.92rem !important; padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important; padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

hr { border-color: #1e2535 !important; }
.chat-container {
    max-height: 60vh; overflow-y: auto; padding: 0.5rem 0.2rem;
    scrollbar-width: thin; scrollbar-color: #1e2535 transparent;
}
[data-testid="stFileUploader"] {
    background: #131720; border: 1px dashed #1e2535; border-radius: 10px; padding: 0.5rem;
}
.empty-state { text-align: center; margin-top: 4rem; color: #334155; }
.empty-state .icon { font-size: 4rem; margin-bottom: 1rem; }
.empty-state h3 { font-size: 1.1rem; font-weight: 600; color: #475569; margin-bottom: 0.5rem; }

.user-pill {
    display: inline-block; background: #0f172a; border: 1px solid #334155;
    border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────
for key, default in [
    ("user_id",        None),
    ("username",       None),
    ("email",          None),
    ("session_id",     None),   # current DB session
    ("messages",       []),
    ("chat_history",   []),
    ("chain",          None),
    ("loaded_sources", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────
#  HISTORY NORMALIZER
# ─────────────────────────────────────────────────────────────
def normalize_history(history: list) -> list:
    normalized = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "human":   role = "user"
            elif role == "ai":    role = "assistant"
            normalized.append({"role": role, "content": str(content)})
        elif isinstance(msg, HumanMessage):
            normalized.append({"role": "user",      "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            normalized.append({"role": "assistant", "content": str(msg.content)})
        elif isinstance(msg, SystemMessage):
            normalized.append({"role": "system",    "content": str(msg.content)})
        elif hasattr(msg, "content"):
            normalized.append({"role": "user",      "content": str(msg.content)})
        else:
            normalized.append({"role": "user",      "content": str(msg)})
    return normalized


# ─────────────────────────────────────────────────────────────
#  RAW LLM CALLER
# ─────────────────────────────────────────────────────────────
def call_llm(messages: list, model: str) -> str:
    client = openai.OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://multi-rag.local",
            "X-Title":      "Multi-Source RAG",
        },
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        temperature=0.3,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


# ── Embeddings ───────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Loaders ──────────────────────────────────
def load_from_url(url: str):
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {"timeout": 15}
    docs = loader.load()
    title = docs[0].metadata.get("title", url) if docs else url
    return docs, title

def load_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    docs = PyPDFLoader(tmp_path).load()
    os.unlink(tmp_path)
    return docs, uploaded_file.name

def load_from_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": uploaded_file.name})], uploaded_file.name

def load_from_csv(uploaded_file, batch_size=2):
    import csv
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    rows = list(csv.DictReader(io.StringIO(raw)))
    docs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        lines = [", ".join(f"{k}: {v}" for k, v in row.items() if v.strip()) for row in batch]
        docs.append(Document(
            page_content="\n".join(lines),
            metadata={"source": uploaded_file.name, "rows": f"{i}-{i+len(batch)-1}"}
        ))
    return docs, uploaded_file.name

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────────────────────
#  BUILD CHAIN
# ─────────────────────────────────────────────────────────────
def build_chain(docs, model: str):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    def retrieve(query: str):
        return vectorstore.similarity_search(query, k=7)

    REWRITE_SYSTEM = (
        "You are a query rewriter. Given a conversation history and a follow-up question, "
        "rewrite the follow-up as a fully self-contained standalone question that captures "
        "all necessary context from the history. Output ONLY the rewritten question, nothing else."
    )

    ANSWER_SYSTEM = (
        "You are an expert assistant with deep knowledge of movies and TV shows. "
        "You are given content retrieved from a Netflix dataset and the full conversation history.\n\n"
        "Rules:\n"
        "- Use the conversation history to maintain full contextual awareness across turns.\n"
        "- If the user refers to something mentioned earlier (e.g. 'the first one', 'that movie', "
        "'tell me more'), resolve it using the history.\n"
        "- For recommendations: prioritize titles with compelling descriptions and explain "
        "WHY each title is worth watching in 1 sentence.\n"
        "- For specific lookups: answer directly and precisely.\n"
        "- For counting questions: clarify you only see partial data.\n"
        "- Always keep your answer grounded in the retrieved content."
    )

    def run_chain(question: str, chat_history) -> str:
        history_dicts = normalize_history(chat_history if chat_history else [])

        standalone_question = question
        if history_dicts:
            rewrite_messages = (
                [{"role": "system", "content": REWRITE_SYSTEM}]
                + history_dicts
                + [{"role": "user", "content": question}]
            )
            standalone_question = call_llm(rewrite_messages, model)

        context = format_docs(retrieve(standalone_question))

        answer_messages = (
            [{"role": "system", "content": ANSWER_SYSTEM}]
            + history_dicts
            + [{
                "role": "user",
                "content": (
                    f"Retrieved content:\n{context}\n\n"
                    f"Question: {standalone_question}"
                )
            }]
        )
        return call_llm(answer_messages, model)

    return run_chain, len(chunks)


# ─────────────────────────────────────────────────────────────
#  LOGIN GATE  (shown when no user is logged in)
# ─────────────────────────────────────────────────────────────
def render_login():
    st.markdown("""
    <div class="app-header">
        <h1>🤖 Multi-Source RAG Chatbot</h1>
        <p>Sign in to save and resume your chat history across sessions</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("## 👋 Welcome back")
        st.markdown(
            '<p>Enter your email and username. A new account is created automatically '
            'if you\'re new — returning users will have their history restored.</p>',
            unsafe_allow_html=True,
        )

        email    = st.text_input("Email address",   placeholder="you@example.com",  key="login_email")
        username = st.text_input("Display name",    placeholder="Your name",         key="login_username")

        if st.button("Continue →", use_container_width=True):
            if not email.strip() or not username.strip():
                st.error("Please fill in both fields.")
            elif "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                uid = get_or_create_user(email.strip(), username.strip())
                st.session_state.user_id  = uid
                st.session_state.username = username.strip()
                st.session_state.email    = email.strip().lower()
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MAIN APP  (shown after login)
# ─────────────────────────────────────────────────────────────
def render_app():
    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        # User identity
        st.markdown(
            f"<p style='margin-bottom:4px; color:#64748b; font-size:0.78rem'>Signed in as</p>"
            f"<span class='user-pill'>👤 {st.session_state.username}</span>"
            f"<span style='color:#475569; font-size:0.72rem; margin-left:6px'>{st.session_state.email}</span>",
            unsafe_allow_html=True,
        )
        if st.button("Sign out", use_container_width=True, key="signout"):
            for k in ["user_id", "username", "email", "session_id",
                      "messages", "chat_history", "chain", "loaded_sources"]:
                st.session_state[k] = None if k in ("user_id", "session_id") else (
                    [] if k in ("messages", "chat_history", "loaded_sources") else None
                )
            st.rerun()

        st.markdown("---")

        # ── Past sessions ──────────────────────
        sessions = get_user_sessions(st.session_state.user_id)
        if sessions:
            st.markdown("### 🕓 Your Sessions")
            for s in sessions:
                meta = json.loads(s["source_meta"]) if s["source_meta"] else {}
                source_label = f"{meta.get('icon','')} {meta.get('name','Unknown source')}" if meta else "No source"
                col_s, col_d = st.columns([5, 1])
                with col_s:
                    if st.button(
                        f"📂 {s['session_tag']}\n{source_label}",
                        key=f"sess_{s['id']}",
                        use_container_width=True,
                    ):
                        # Load this session's messages
                        msgs = get_session_messages(s["id"])
                        st.session_state.session_id    = s["id"]
                        st.session_state.messages      = msgs
                        st.session_state.chat_history  = msgs.copy()
                        st.session_state.chain         = None   # user must re-load source
                        st.session_state.loaded_sources = [meta] if meta else []
                        st.rerun()
                with col_d:
                    if st.button("🗑", key=f"del_{s['id']}", help="Delete this session"):
                        delete_session(s["id"])
                        if st.session_state.session_id == s["id"]:
                            st.session_state.session_id   = None
                            st.session_state.messages     = []
                            st.session_state.chat_history = []
                            st.session_state.chain        = None
                            st.session_state.loaded_sources = []
                        st.rerun()

            if st.button("➕ New session", use_container_width=True):
                sid = create_session(st.session_state.user_id)
                st.session_state.session_id    = sid
                st.session_state.messages      = []
                st.session_state.chat_history  = []
                st.session_state.chain         = None
                st.session_state.loaded_sources = []
                st.rerun()

            st.markdown("---")

        # ── Model picker ──────────────────────
        st.markdown("## ⚙️ Settings")
        model = st.selectbox(
            "Model",
            options=[
                "arcee-ai/trinity-large-preview:free",
                "deepseek/deepseek-r1-0528:free",
                "nvidia/nemotron-3-nano-30b-a3b:free",
                "openai/gpt-4o-mini",
            ],
            index=0,
        )
        st.markdown(
            "<p style='color:#38bdf8; font-size:0.75rem'>🔷 Embeddings: all-MiniLM-L6-v2</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Source loader ─────────────────────
        st.markdown("### 📂 Add Source")
        source_type = st.radio(
            "Source type",
            options=["🌐 Website", "📄 PDF", "📝 Text File", "📊 CSV"],
            label_visibility="collapsed",
        )

        def _after_load(docs, title, icon, detail):
            """Common logic after any source is loaded."""
            chain, n_chunks = build_chain(docs, model)
            src_meta = {"icon": icon, "name": title, "detail": detail, "chunks": n_chunks}

            # Create a new DB session if none active
            if not st.session_state.session_id:
                sid = create_session(st.session_state.user_id, src_meta)
                st.session_state.session_id = sid
            else:
                update_session_source(st.session_state.session_id, src_meta)

            st.session_state.chain          = chain
            st.session_state.messages       = []
            st.session_state.chat_history   = []
            st.session_state.loaded_sources = [src_meta]
            st.rerun()

        if source_type == "🌐 Website":
            url_input = st.text_input("URL", placeholder="https://example.com/article", label_visibility="collapsed")
            if st.button("📥 Load URL", use_container_width=True):
                if not url_input.strip():
                    st.sidebar.error("Please enter a URL.")
                else:
                    with st.spinner("Fetching & embedding..."):
                        try:
                            docs, title = load_from_url(url_input.strip())
                            _after_load(docs, title, "🌐", url_input.strip())
                        except Exception as e:
                            st.sidebar.error(f"Error: {e}")

        elif source_type == "📄 PDF":
            pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
            if st.button("📥 Load PDF", use_container_width=True):
                if not pdf_file:
                    st.sidebar.error("Please upload a PDF file.")
                else:
                    with st.spinner("Parsing & embedding..."):
                        try:
                            docs, title = load_from_pdf(pdf_file)
                            _after_load(docs, title, "📄", f"{len(docs)} pages")
                        except Exception as e:
                            st.sidebar.error(f"Error: {e}")

        elif source_type == "📝 Text File":
            txt_file = st.file_uploader("Upload Text File", type=["txt", "md"], label_visibility="collapsed")
            if st.button("📥 Load File", use_container_width=True):
                if not txt_file:
                    st.sidebar.error("Please upload a text file.")
                else:
                    with st.spinner("Reading & embedding..."):
                        try:
                            docs, title = load_from_txt(txt_file)
                            _after_load(docs, title, "📝", f"{len(docs[0].page_content)} chars")
                        except Exception as e:
                            st.sidebar.error(f"Error: {e}")

        elif source_type == "📊 CSV":
            csv_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
            if st.button("📥 Load CSV", use_container_width=True):
                if not csv_file:
                    st.sidebar.error("Please upload a CSV file.")
                else:
                    with st.spinner("Parsing & embedding..."):
                        try:
                            docs, title = load_from_csv(csv_file)
                            _after_load(docs, title, "📊", f"{len(docs)} rows")
                        except Exception as e:
                            st.sidebar.error(f"Error: {e}")

        if st.session_state.loaded_sources:
            st.markdown("---")
            st.markdown("### ✅ Loaded Source")
            for src in st.session_state.loaded_sources:
                st.markdown(f"""
                <div class="doc-card">
                    <strong>{src.get('icon','')} {src.get('name','')}</strong>
                    {src.get('detail','')}<br/>
                    <span class="badge">{src.get('chunks',0)} chunks</span>
                </div>
                """, unsafe_allow_html=True)
            if st.button("🗑️ Clear Source", use_container_width=True):
                st.session_state.chain          = None
                st.session_state.loaded_sources = []
                st.rerun()

        st.markdown("---")
        st.markdown(
            "<p style='color:#334155; font-size:0.78rem; text-align:center'>Built with LangChain + OpenRouter</p>",
            unsafe_allow_html=True,
        )

    # ── Main chat area ─────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>🤖 Multi-Source RAG Chatbot</h1>
        <p>Load a Website · PDF · Text File · CSV — then ask anything about it</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chain:
        # Show history if a session is loaded but source isn't re-embedded yet
        if st.session_state.messages:
            st.info(
                "📂 Previous conversation loaded. Re-load the source in the sidebar to continue chatting.",
                icon="ℹ️",
            )
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for msg in st.session_state.messages:
                css_cls = "chat-user" if msg["role"] == "user" else "chat-bot"
                st.markdown(
                    f'<div class="{css_cls}"><div class="bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">📂</div>
                <h3>No source loaded yet</h3>
                <p>Choose a source type in the sidebar, upload or paste a URL, and click Load.</p>
                <br/>
                <p style="color:#1e2535; font-size:0.82rem">
                    Supports &nbsp;🌐 Websites &nbsp;·&nbsp; 📄 PDFs &nbsp;·&nbsp; 📝 Text files &nbsp;·&nbsp; 📊 CSV files
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            css_cls = "chat-user" if msg["role"] == "user" else "chat-bot"
            st.markdown(
                f'<div class="{css_cls}"><div class="bubble">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask a question",
                placeholder="Ask anything about the loaded source...",
                label_visibility="collapsed",
                key="question_input",
            )
        with col2:
            send = st.button("Send ➤", use_container_width=True)

        if send and user_input.strip():
            question = user_input.strip()
            st.session_state.messages.append({"role": "user", "content": question})

            # Ensure a DB session exists
            if not st.session_state.session_id:
                sid = create_session(
                    st.session_state.user_id,
                    st.session_state.loaded_sources[0] if st.session_state.loaded_sources else None,
                )
                st.session_state.session_id = sid

            # Persist user message
            save_message(st.session_state.session_id, "user", question)

            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain(
                        question,
                        st.session_state.chat_history,
                    )
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Persist assistant message
                    save_message(st.session_state.session_id, "assistant", answer)

                    st.session_state.chat_history.append({"role": "user",      "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history = st.session_state.chat_history[-20:]

                except Exception as e:
                    err_msg = f"⚠️ Error: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    save_message(st.session_state.session_id, "assistant", err_msg)

            st.rerun()


# ─────────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────────
if st.session_state.user_id is None:
    render_login()
else:
    render_app()