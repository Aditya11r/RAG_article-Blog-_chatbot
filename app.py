import os
import io
import tempfile
import streamlit as st
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert assistant with deep knowledge of movies and TV shows.\n"
     "You are given content retrieved from a Netflix dataset.\n\n"
     "Rules:\n"
     "- For recommendations: prioritize titles with compelling descriptions.\n"
     "- Always explain WHY each title is worth watching in 1 sentence.\n"
     "- For specific lookups: answer directly and precisely.\n"
     "- For counting questions: clarify you only see partial data.\n"
    ),
    MessagesPlaceholder("chat_history"),   # ✅ ADD THIS
    ("human", "Content:\n{context}\n\nQuestion: {question}")
])
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if not OPENROUTER_API_KEY:
    st.error("❌ OPENROUTER_API_KEY not found.")
    st.stop()



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
</style>
""", unsafe_allow_html=True)

for key, default in [
    ("messages", []),
    ("chat_history", []),   # ✅ ADD THIS
    ("chain", None),
    ("loaded_sources", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True}
    )

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
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    return docs, uploaded_file.name

def load_from_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
    return docs, uploaded_file.name

def load_from_csv(uploaded_file, batch_size=2):
    import csv
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(raw))
    rows = list(reader)
    docs = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        lines = []
        for row in batch:
            row_text = ", ".join(f"{k}: {v}" for k, v in row.items() if v.strip())
            lines.append(row_text)
        text = chr(10).join(lines)
        docs.append(Document(
            page_content=text,
            metadata={"source": uploaded_file.name, "rows": f"{i}-{i+len(batch)-1}"}
        ))
    return docs, uploaded_file.name

# ─────────────────────────────────────────────
#  Build RAG chain
# ─────────────────────────────────────────────
def build_chain(docs, model: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    llm = ChatOpenAI(
        model=model,
        max_tokens=500,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
        default_headers={
            "HTTP-Referer": "https://multi-rag.local",
            "X-Title": "Multi-Source RAG",
        },
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question "
         "which might reference context in the chat history, "
         "formulate a standalone question that can be understood "
         "without the chat history. Do NOT answer the question."
         ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    # ✅ FIX: Use RunnableLambda to safely extract fields before prompt
    
    def rewrite_question(inputs: dict) -> str:
        question = inputs["question"]
        chat_history = inputs["chat_history"]
        if not chat_history:
            return question
        messages = contextualize_q_prompt.format_messages(
        chat_history=chat_history,
        question=question
    )
        return llm.invoke(messages).content

    history_aware_retriever = (
    RunnableLambda(rewrite_question)
    | base_retriever
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert assistant with deep knowledge of movies and TV shows.\n"
     "You are given content retrieved from a Netflix dataset.\n\n"
     "Rules:\n"
     "- For recommendations: prioritize titles with compelling descriptions.\n"
     "- Always explain WHY each title is worth watching in 1 sentence.\n"
     "- For specific lookups: answer directly and precisely.\n"
     "- For counting questions: clarify you only see partial data.\n"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "Content:\n{context}\n\nQuestion: {question}")
    ])

# ✅ FIX: Don't pass the dict directly into the chain.
# Build a RunnableLambda that manually invokes prompt + llm
    def run_chain(inputs: dict) -> str:
        question = inputs["question"]
        chat_history = inputs["chat_history"]
    
    # Get context via retriever
        context_docs = history_aware_retriever.invoke({
        "question": question,
        "chat_history": chat_history
    })
        context = format_docs(context_docs)
    
    # Build messages manually — no ambiguity
        messages = prompt.format_messages(
        chat_history=chat_history,
        context=context,
        question=question
    )
    
        response = llm.invoke(messages)
        return response.content

    chain = RunnableLambda(run_chain)
    return chain, len(chunks)

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    model = st.selectbox(
        "Model",
        options=[
            "arcee-ai/trinity-large-preview:free",
            "deepseek/deepseek-r1-0528:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "openai/gpt-4o-mini"
            
        ],
        index=0,
    )

    st.markdown(
        "<p style='color:#38bdf8; font-size:0.75rem'>🔷 Embeddings:free: all-MiniLM-L6-v2</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📂 Add Source")

    source_type = st.radio(
        "Source type",
        options=["🌐 Website", "📄 PDF", "📝 Text File", "📊 CSV"],
        label_visibility="collapsed",
    )

    # Website
    if source_type == "🌐 Website":
        url_input = st.text_input("URL", placeholder="https://example.com/article", label_visibility="collapsed")
        if st.button("📥 Load URL", use_container_width=True):
            if not url_input.strip():
                st.sidebar.error("Please enter a URL.")
            else:
                with st.spinner("Fetching & embedding..."):
                    try:
                        docs, title = load_from_url(url_input.strip())
                        chain, n_chunks = build_chain(docs, model)
                        st.session_state.chain = chain
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.loaded_sources = [{"icon": "🌐", "name": title, "detail": url_input.strip(), "chunks": n_chunks}]
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error: {e}")

    # PDF
    elif source_type == "📄 PDF":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if st.button("📥 Load PDF", use_container_width=True):
            if not pdf_file:
                st.sidebar.error("Please upload a PDF file.")
            else:
                with st.spinner("Parsing & embedding..."):
                    try:
                        docs, title = load_from_pdf(pdf_file)
                        chain, n_chunks = build_chain(docs, model)
                        st.session_state.chain = chain
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.loaded_sources = [{"icon": "📄", "name": title, "detail": f"{len(docs)} pages", "chunks": n_chunks}]
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error: {e}")

    # Text File
    elif source_type == "📝 Text File":
        txt_file = st.file_uploader("Upload Text File", type=["txt", "md"], label_visibility="collapsed")
        if st.button("📥 Load File", use_container_width=True):
            if not txt_file:
                st.sidebar.error("Please upload a text file.")
            else:
                with st.spinner("Reading & embedding..."):
                    try:
                        docs, title = load_from_txt(txt_file)
                        chain, n_chunks = build_chain(docs, model)
                        st.session_state.chain = chain
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.loaded_sources = [{"icon": "📝", "name": title, "detail": f"{len(docs[0].page_content)} chars", "chunks": n_chunks}]
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error: {e}")

    # CSV
    elif source_type == "📊 CSV":
        csv_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if st.button("📥 Load CSV", use_container_width=True):
            if not csv_file:
                st.sidebar.error("Please upload a CSV file.")
            else:
                with st.spinner("Parsing & embedding..."):
                    try:
                        docs, title = load_from_csv(csv_file)
                        chain, n_chunks = build_chain(docs, model)
                        st.session_state.chain = chain
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.loaded_sources = [{"icon": "📊", "name": title, "detail": f"{len(docs)} rows", "chunks": n_chunks}]
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error: {e}")

    # Loaded source info
    if st.session_state.loaded_sources:
        st.markdown("---")
        st.markdown("### ✅ Loaded Source")
        for src in st.session_state.loaded_sources:
            st.markdown(f"""
            <div class="doc-card">
                <strong>{src['icon']} {src['name']}</strong>
                {src['detail']}<br/>
                <span class="badge">{src['chunks']} chunks</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.chain = None
            st.session_state.loaded_sources = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color:#334155; font-size:0.78rem; text-align:center'>Built with LangChain + OpenRouter</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Main chat area
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🤖 Multi-Source RAG Chatbot</h1>
    <p>Load a Website · PDF · Text File · CSV — then ask anything about it</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.chain:
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
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><div class="bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot"><div class="bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
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

        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.chain.invoke({
                    "question": question,
                    "chat_history": st.session_state.chat_history
                })

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                # ✅ Proper BaseMessage history
                st.session_state.chat_history.append(
                    HumanMessage(content=question)
                )
                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )

                # keep last 10 messages
                st.session_state.chat_history = st.session_state.chat_history[-10:]

            except Exception as e:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ Error: {e}"}
                )

        st.rerun()