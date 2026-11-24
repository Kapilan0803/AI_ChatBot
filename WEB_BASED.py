import streamlit as st

from utils import (
    build_retriever,
    explain_error,
    key_missing_message,
    load_web_page,
    normalize_url,
    provider_ready,
    stream_rag_answer,
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        color: #e6205b;
        font-size: 34px;
        font-weight: bold;
        border: 3px solid #e6205b;
        padding: 8px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">🌐✨ WEB-Knowledge Chatbot</h1>', unsafe_allow_html=True)

# ── Session state ──
st.session_state.setdefault("web_messages", [])
st.session_state.setdefault("web_retriever", None)
st.session_state.setdefault("web_loaded_url", None)

with st.sidebar:
    if st.button("🗑️ Clear Chat History", key="clear_web"):
        st.session_state["web_messages"] = []
        st.rerun()

# ── Input for website URL ──
col1, col2 = st.columns([4, 1])
with col1:
    url_input = st.text_input(
        "Enter a website URL to extract knowledge:",
        placeholder="e.g., https://en.wikipedia.org/wiki/Artificial_intelligence",
        label_visibility="collapsed",
    )
with col2:
    load_btn = st.button("🚀 Load URL")

url = normalize_url(url_input)

if load_btn and url:
    with st.spinner("⚡ Fetching and indexing web page..."):
        try:
            docs = load_web_page(url)
            if not docs:
                raise ValueError(
                    "Could not extract readable text from this page "
                    "(it may be rendered by JavaScript or blocked for bots)."
                )
            retriever, n_chunks = build_retriever(docs)
        except Exception as e:
            # Reset, so the same URL can be retried and an old index is never reused by mistake.
            st.session_state["web_retriever"] = None
            st.session_state["web_loaded_url"] = None
            st.error(f"Error loading URL: {e}")
        else:
            st.session_state["web_retriever"] = retriever
            st.session_state["web_loaded_url"] = url  # only after success
            st.session_state["web_messages"] = []
            st.success(f"✅ Ready! Indexed `{url}` ({n_chunks} chunks).")
elif load_btn:
    st.warning("Please enter a URL first.")
elif st.session_state["web_retriever"] is not None:
    st.caption(f"🌐 Active page: `{st.session_state['web_loaded_url']}`")

# ── Display existing chat history ──
for msg in st.session_state["web_messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Query input ──
if st.session_state["web_retriever"] is not None:
    if user_query := st.chat_input("Ask a question about the webpage..."):
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            if not provider_ready():
                st.error(key_missing_message())
            else:
                try:
                    answer = st.write_stream(
                        stream_rag_answer(st.session_state["web_retriever"], user_query)
                    )
                except Exception as e:
                    st.error(explain_error(e))
                else:
                    st.session_state["web_messages"].append({"role": "user", "content": user_query})
                    st.session_state["web_messages"].append({"role": "assistant", "content": answer})
else:
    st.info("👆 Enter a URL above and click **Load URL** to chat with any web page.")
