import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.logo("Chatbots-removebg-preview.png")

# ── Helper to format retrieved document chunks ──
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

    if st.button("🗑️ Clear Chat History"):
        st.session_state["rag_messages"] = []
        st.rerun()

active_key = api_key_input or os.getenv("GROQ_API_KEY", "")

# ── Session state setup ──
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None
if "rag_file_id" not in st.session_state:
    st.session_state["rag_file_id"] = None

# File uploader
uploaded_file = st.file_uploader(
    "Upload a document (PDF or Text):", type=["pdf", "txt"], help="Fast indexing powered by in-memory embeddings."
)

if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state["rag_file_id"] != file_id:
        st.session_state["rag_file_id"] = file_id
        st.session_state["rag_chain"] = None
        st.session_state["rag_messages"] = []

        with st.spinner("⚡ Fast indexing document chunks..."):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()
                else:
                    loader = TextLoader(temp_file_path, encoding="utf-8")
                    docs = loader.load()

                if not docs:
                    st.error("The document is empty.")
                    st.stop()

                # Optimized chunking for speed
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                final_documents = text_splitter.split_documents(docs[:60])

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
                    """Answer the question based only on the following context. Be concise, direct, and factual.
<context>
{context}
</context>
Question: {input}"""
                )

                retriever = vectors.as_retriever(search_kwargs={"k": 3})

                # Modern LCEL RAG chain
                st.session_state["rag_chain"] = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                st.success(f"✅ Ready! Indexed {len(final_documents)} chunks from `{uploaded_file.name}`.")

            except Exception as e:
                st.error(f"Error indexing document: {e}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# Display existing chat history
for msg in st.session_state["rag_messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Query input
if st.session_state["rag_chain"]:
    if user_query := st.chat_input("Ask a question about your document..."):
        st.session_state["rag_messages"].append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        if not active_key or active_key == "your_api_key":
            with st.chat_message("assistant"):
                st.error("⚠️ Please provide a valid Groq API Key in the sidebar.")
        else:
            with st.chat_message("assistant"):
                try:
                    # Stream tokens in real-time
                    chain = st.session_state["rag_chain"]
                    answer = st.write_stream(chain.stream(user_query))
                    st.session_state["rag_messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
elif not uploaded_file:
    st.info("👆 Upload a PDF or Text file above to start questioning your documents.")
