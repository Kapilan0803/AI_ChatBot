import streamlit as st

# Custom CSS for the home page
st.markdown(
    """
    <style>
    .premium-header {
        color: #c45b10;
        font-size: 36px;
        font-weight: bold;
        border: 4px solid #c45b10;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #e30780;
        font-size: 28px;
        text-align: center;
        margin-top: 28px;
        font-family: cursive;
    }
    .description {
        color: #eb7907;
        font-size: 22px;
        text-align: justify;
        margin: 20px auto;
        font-family: cursive;
        line-height: 2.0;
    }
    .social-links {
        margin-top: 30px;
        text-align: center;
    }
    .social-links a {
        color: royalblue;
        text-decoration: none;
        font-size: 24px;
        margin: 0 15px;
    }
    .social-links a:hover {
        color: #e30780;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Premium header title
st.markdown('<div class="premium-header">AI Chatbot Suite</div>', unsafe_allow_html=True)

# Subtitle with custom color
st.markdown(
    '<div class="subtitle">Conversational, Document and Web-based AI Solutions</div>',
    unsafe_allow_html=True,
)

# Brief description
st.markdown(
    '<div class="description">Welcome to the AI Chatbot Suite, an elite platform at the '
    "forefront of conversational AI innovation. Whether you desire effortless, human-like "
    "interactions through our cutting-edge Conversational Chatbot, data-enriched responses "
    "driven by our advanced Retrieval-Augmented Generation (RAG)-based Chatbot, or web-based "
    "knowledge extraction, we are here to redefine your digital experience and elevate "
    "engagement to new heights.</div>",
    unsafe_allow_html=True,
)

st.markdown("")

# Feature cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 💬 ConvoAI")
    st.write("Natural conversational chatbot powered by Google Gemini (free), Groq or local Ollama models.")
with col2:
    st.markdown("### 📚 RAG Chatbot")
    st.write("Upload PDFs and ask questions — answers grounded in your documents.")
with col3:
    st.markdown("### 🌐 Web RAG")
    st.write("Enter any URL and chat about its content instantly.")

st.info("👈 Pick a tool from the sidebar. Choose an AI provider and add its free API key in the sidebar settings (Ollama needs no key).")

st.divider()

st.markdown(
    """
    <div class="social-links">
        <a href="https://www.linkedin.com/******************" target="_blank">LinkedIn</a>
        <a href="https://github.com/************" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)

