import streamlit as st

from utils import (
    build_retriever,
    explain_error,
    key_missing_message,
    load_uploaded_document,
    provider_ready,
    stream_rag_answer,
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        color: #4CAF50;
        font-size: 34px;
        font-weight: bold;
        border: 3px solid #4CAF50;
        padding: 8px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">📚🔍 RAG-based Chatbot</h1>', unsafe_allow_html=True)

# ── Session state ──
st.session_state.setdefault("rag_messages", [])
st.session_state.setdefault("rag_retriever", None)
st.session_state.setdefault("rag_file_id", None)
st.session_state.setdefault("rag_file_name", None)

with st.sidebar:
    if st.button("🗑️ Clear Chat History", key="clear_rag"):
        st.session_state["rag_messages"] = []
        st.rerun()

# ── File uploader ──
uploaded_file = st.file_uploader(
    "Upload a document (PDF or Text):",
    type=["pdf", "txt"],
    help="The whole document is indexed in memory with local embeddings.",
)

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state["rag_file_id"] != file_id:
        with st.spinner("⚡ Reading and indexing your document..."):
            try:
                docs = load_uploaded_document(uploaded_file)
                retriever, n_chunks = build_retriever(docs)
            except Exception as e:
                # Reset, so the same file can be retried and an old index is never reused by mistake.
                st.session_state["rag_retriever"] = None
                st.session_state["rag_file_id"] = None
                st.error(f"Could not index this document: {e}")
            else:
                st.session_state["rag_retriever"] = retriever
                st.session_state["rag_file_id"] = file_id  # only after success
                st.session_state["rag_file_name"] = uploaded_file.name
                st.session_state["rag_messages"] = []
                st.success(f"✅ Ready! Indexed {n_chunks} chunks from `{uploaded_file.name}`.")
elif st.session_state["rag_retriever"] is not None:
    st.caption(
        f"📄 Active document: `{st.session_state['rag_file_name']}` — upload a new file to replace it."
    )

# ── Display existing chat history ──
for msg in st.session_state["rag_messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Query input ──
if st.session_state["rag_retriever"] is not None:
    if user_query := st.chat_input("Ask a question about your document..."):
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            if not provider_ready():
                st.error(key_missing_message())
            else:
                try:
                    answer = st.write_stream(
                        stream_rag_answer(st.session_state["rag_retriever"], user_query)
                    )
                except Exception as e:
                    st.error(explain_error(e))
                else:
                    st.session_state["rag_messages"].append({"role": "user", "content": user_query})
                    st.session_state["rag_messages"].append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload a PDF or Text file above to start questioning your documents.")
