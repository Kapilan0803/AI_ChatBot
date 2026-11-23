import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.logo("Chatbots-removebg-preview.png")

# ── Cache Embeddings model in RAM across sessions for maximum speed ──
@st.cache_resource(show_spinner="⚡ Loading fast embedding engine into memory (one-time)...")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
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

# Sidebar API Key & Settings
env_key = os.getenv("GROQ_API_KEY", "")
if env_key == "your_api_key":
    env_key = ""

with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key_input = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Get your free key at https://console.groq.com",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

active_key = api_key_input or os.getenv("GROQ_API_KEY", "")

# ── Session state setup ──
if "web_messages" not in st.session_state:
    st.session_state["web_messages"] = []
if "web_retrieval_chain" not in st.session_state:
    st.session_state["web_retrieval_chain"] = None
if "web_loaded_url" not in st.session_state:
    st.session_state["web_loaded_url"] = None

# Input for website URL
col1, col2 = st.columns([4, 1])
with col1:
    url = st.text_input(
        "Enter a website URL to extract knowledge:",
        placeholder="e.g., https://en.wikipedia.org/wiki/Artificial_intelligence",
        label_visibility="collapsed",
    )
with col2:
    load_btn = st.button("🚀 Load URL", use_container_width=True)

if load_btn and url:
    if st.session_state["web_loaded_url"] != url.strip():
        st.session_state["web_loaded_url"] = url.strip()
        st.session_state["web_retrieval_chain"] = None
        st.session_state["web_messages"] = []

        with st.spinner("⚡ Fetching and indexing web page..."):
            try:
                loader = WebBaseLoader(url.strip())
                docs = loader.load()

                if not docs or not docs[0].page_content.strip():
                    st.error("Could not extract readable text from this URL.")
                    st.stop()

                # Optimized chunking for speed
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                final_documents = text_splitter.split_documents(docs[:50])

                # Cached embedding model
                embeddings = get_embedding_model()

                # Vector store
                vectors = Chroma.from_documents(final_documents, embeddings)

                llm = ChatGroq(
                    groq_api_key=active_key if active_key and active_key != "your_api_key" else "dummy",
                    model_name="llama-3.1-8b-instant",
                    streaming=True,
                    temperature=0.2,
                )

                prompt = ChatPromptTemplate.from_template(
                    """Answer the question based only on the following webpage context. Be concise and accurate.
<context>
{context}
</context>
Question: {input}"""
                )

                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = vectors.as_retriever(search_kwargs={"k": 3})
                st.session_state["web_retrieval_chain"] = create_retrieval_chain(retriever, document_chain)
                st.success(f"✅ Ready! Indexed `{url.strip()}` ({len(final_documents)} chunks).")

            except Exception as e:
                st.error(f"Error loading URL: {e}")

# Display existing chat history
for msg in st.session_state["web_messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Query input
if st.session_state["web_retrieval_chain"]:
    if user_query := st.chat_input("Ask a question about the webpage..."):
        st.session_state["web_messages"].append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        if not active_key or active_key == "your_api_key":
            with st.chat_message("assistant"):
                st.error("⚠️ Please provide a valid Groq API Key in the sidebar.")
        else:
            with st.chat_message("assistant"):
                try:
                    # Stream chunks in real-time
                    def stream_web():
                        chain = st.session_state["web_retrieval_chain"]
                        for chunk in chain.stream({"input": user_query}):
                            if "answer" in chunk:
                                yield chunk["answer"]

                    answer = st.write_stream(stream_web())
                    st.session_state["web_messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
elif not url:
    st.info("👆 Enter a URL above and click **Load URL** to chat with any web page.")
