"""Shared helpers for the Streamlit pages (settings, loading, indexing, RAG answers)."""
import contextlib
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

import llm

# WebBaseLoader reads USER_AGENT when it is imported, so set it before the first import.
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "Chatbots-removebg-preview.png"


# ───────────────────────── Settings (provider, API key, model) ─────────────────────────
def get_provider() -> str:
    return st.session_state.get("provider") or llm.DEFAULT_PROVIDER


def get_api_key() -> str:
    """Key typed in the sidebar (this browser session only), falling back to .env."""
    provider = get_provider()
    return llm.get_api_key(provider, st.session_state.get(f"api_key_{provider}"))


def get_model() -> str:
    provider = get_provider()
    return (st.session_state.get(f"model_{provider}") or "").strip() or llm.default_model(provider)


def provider_ready() -> bool:
    provider = get_provider()
    return llm.is_ready(provider, st.session_state.get(f"api_key_{provider}"))


def key_missing_message() -> str:
    return llm.key_missing_message(get_provider())


def explain_error(exc: Exception) -> str:
    return llm.friendly_error(exc, get_provider())


def render_sidebar_settings() -> None:
    """Called once from main.py so the settings are shared by every page."""
    # Streamlit forgets widgets that are not drawn on a run. Re-saving their values keeps
    # each provider's key/model when the user switches provider and back again.
    for name in llm.PROVIDERS:
        for key in (f"api_key_{name}", f"model_{name}"):
            if key in st.session_state:
                st.session_state[key] = st.session_state[key]

    names = list(llm.PROVIDERS)
    with st.sidebar:
        st.subheader("⚙️ Settings")
        provider = st.selectbox(
            "AI provider",
            names,
            index=names.index(llm.DEFAULT_PROVIDER),
            format_func=lambda n: llm.PROVIDERS[n]["label"],
            key="provider",
        )
        cfg = llm.PROVIDERS[provider]

        if llm.needs_key(provider):
            st.text_input(
                f"{cfg['short']} API key",
                type="password",
                key=f"api_key_{provider}",
                placeholder="Using key from .env" if llm.get_api_key(provider) else "paste your key here",
                help=f"Get a free key at {cfg['key_url']}",
            )

        if provider == "ollama":
            st.session_state.setdefault(f"model_{provider}", llm.default_model(provider))
            st.text_input(
                "Model (must be downloaded with `ollama pull`)",
                key=f"model_{provider}",
            )
        else:
            options = list(dict.fromkeys([llm.default_model(provider), *cfg["models"]]))
            st.selectbox("Model", options, key=f"model_{provider}")


# ───────────────────────────── Embeddings & indexing ─────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading embedding model (the first run downloads ~90 MB)...")
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_retriever(docs: list[Document], k: int = 4):
    """Split → embed → index. Each call builds its own fresh in-memory index, so
    documents/pages never leak into each other. Returns (retriever, chunk_count)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No readable text found. If this is a PDF it may be a scanned/image-only "
            "file (it needs OCR first); if it is a web page it may be rendered by JavaScript."
        )

    try:  # FAISS if installed, otherwise fall back to LangChain's built-in store
        import faiss  # noqa: F401
        from langchain_community.vectorstores import FAISS as VectorStore
    except ImportError:
        from langchain_core.vectorstores import InMemoryVectorStore as VectorStore

    store = VectorStore.from_documents(chunks, get_embedding_model())
    return store.as_retriever(search_kwargs={"k": k}), len(chunks)


# ───────────────────────────────── Loaders ─────────────────────────────────
def load_uploaded_document(uploaded_file) -> list[Document]:
    """Read an uploaded PDF or text file into LangChain Documents."""
    name = uploaded_file.name
    data = uploaded_file.getvalue()

    if name.lower().endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            docs = PyPDFLoader(path).load()
        finally:
            with contextlib.suppress(OSError):
                os.remove(path)
        for doc in docs:
            doc.metadata["source"] = name
        return docs

    text = data.decode("utf-8-sig", errors="replace")
    return [Document(page_content=text, metadata={"source": name})]


def normalize_url(raw: str) -> str:
    url = raw.strip()
    if url and not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _tidy_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_web_page(url: str) -> list[Document]:
    """Download a web page and return its cleaned text as Documents."""
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(
        web_paths=[url],
        requests_kwargs={"timeout": 20},
        raise_for_status=True,
    )
    docs = loader.load()
    for doc in docs:
        doc.page_content = _tidy_text(doc.page_content)
    return [d for d in docs if d.page_content]


# ───────────────────────────────── RAG answering ─────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. Be concise, accurate and direct.
If the answer is not in the context, say you could not find it in the provided content.

<context>
{context}
</context>

Question: {input}"""
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def stream_rag_answer(retriever, question: str):
    """Retrieve the best chunks and stream the answer.

    The model is created here, at question time, so it always uses the provider,
    API key and model currently selected in the sidebar (not whatever was set
    when the document was indexed)."""
    chat_model = llm.make_llm(get_provider(), get_api_key(), get_model(), temperature=0.2)
    chain = RAG_PROMPT | chat_model | StrOutputParser()
    context = format_docs(retriever.invoke(question))
    return chain.stream({"context": context, "input": question})
