import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    st.error("❌ OPENROUTER_API_KEY not found. Please add it to your .env file.")
    st.stop()

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="News Article Q&A",
    page_icon="📰",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background: #0d0f14; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: #131720 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.app-header { text-align: center; padding: 2rem 0 1rem 0; }
.app-header h1 {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.app-header p { color: #64748b; font-size: 0.95rem; font-weight: 300; }

.article-card {
    background: #131720; border: 1px solid #1e2535;
    border-left: 3px solid #38bdf8; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 1.2rem;
    font-size: 0.88rem; color: #94a3b8;
}
.article-card strong { color: #e2e8f0; font-size: 0.95rem; }

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
    max-height: 62vh; overflow-y: auto; padding: 0.5rem 0.2rem;
    scrollbar-width: thin; scrollbar-color: #1e2535 transparent;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "article_meta" not in st.session_state:
    st.session_state.article_meta = None

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    model = st.selectbox(
        "Model",
        options=[
            "mistralai/mistral-7b-instruct",
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
            "meta-llama/llama-3-8b-instruct",
            "google/gemma-3-12b-it:free",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🔗 Article URL")
    url_input = st.text_input(
        "Paste a news article URL",
        placeholder="https://example.com/article",
        label_visibility="collapsed",
    )

    load_btn = st.button("📥 Load Article", use_container_width=True)

    if st.session_state.article_meta:
        st.markdown("---")
        st.markdown("### 📄 Loaded Article")
        meta = st.session_state.article_meta
        st.markdown(f"""
        <div class="article-card">
            <strong>{meta['title']}</strong><br/>
            <span style="color:#38bdf8; font-size:0.78rem">{meta['url']}</span><br/><br/>
            {meta['chunks']} chunks indexed
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear & Reset", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chain = None
            st.session_state.article_meta = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color:#334155; font-size:0.78rem; text-align:center'>Built with LangChain + OpenRouter</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  RAG helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain_from_url(url: str, api_key: str, model: str):
    # 1. Load
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {"timeout": 15}
    docs = loader.load()
    title = docs[0].metadata.get("title", url) if docs else url

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed + FAISS
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. LLM
    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "https://news-rag.local",
            "X-Title": "News Article RAG",
        },
    )

    # 5. Prompt
    prompt = PromptTemplate.from_template(
        "You are a news analyst assistant. Use ONLY the article content below to answer.\n"
        "If the answer is not in the article, say 'This information is not covered in the article.'\n"
        "Be concise and factual.\n\n"
        "Article content:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # 6. Modern LCEL chain (no deprecated RetrievalQA)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, title, len(chunks)


# ─────────────────────────────────────────────
#  Load article button
# ─────────────────────────────────────────────
if load_btn:
    if not url_input.strip():
        st.sidebar.error("Please enter a valid URL.")
    else:
        with st.spinner("Fetching and indexing article..."):
            try:
                chain, title, n_chunks = build_chain_from_url(
                    url_input.strip(), OPENROUTER_API_KEY, model
                )
                st.session_state.chain = chain
                st.session_state.article_meta = {
                    "title": title,
                    "url": url_input.strip(),
                    "chunks": n_chunks,
                }
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to load article: {e}")

# ─────────────────────────────────────────────
#  Main chat area
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>📰 News Article-Blog Q&amp;A</h1>
    <p>Paste any news article-Blog URL → ask questions about it</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.chain:
    st.markdown("""
    <div style="text-align:center; margin-top:5rem; color:#334155;">
        <div style="font-size:4rem; margin-bottom:1rem;">🗞️</div>
        <p style="font-size:1.1rem; font-weight:600; color:#475569;">No article-Blog loaded yet</p>
        <p style="font-size:0.88rem;">Paste a news article-Blog URL in the sidebar and click <strong>Load Article</strong></p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user"><div class="bubble">{msg["content"]}</div></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-bot"><div class="bubble">{msg["content"]}</div></div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="What is this article-Blog about? Who is mentioned? What happened?",
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
                answer = st.session_state.chain.invoke(question)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})

        st.rerun()