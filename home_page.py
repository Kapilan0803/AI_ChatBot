import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Suite",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for premium design
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: black;
    }
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
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
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
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.logo("Chatbots-removebg-preview.png")

# Premium header title
st.markdown('<div class="premium-header">AI Chatbot Suite</div>', unsafe_allow_html=True)

# Subtitle with custom color
st.markdown('<div class="subtitle">Conversational and RAG-based AI Solutions</div>', unsafe_allow_html=True)

# Brief description
st.markdown(
    '<div class="description">Welcome to the AI Chatbot Suite, an elite platform at the '
    'forefront of conversational AI innovation. Whether you desire effortless, human-like '
    'interactions through our cutting-edge Conversational Chatbot, data-enriched responses '
    'driven by our advanced Retrieval-Augmented Generation (RAG)-based Chatbot, or web-based '
    'knowledge extraction, we are here to redefine your digital experience and elevate '
    'engagement to new heights.</div>',
    unsafe_allow_html=True
)

st.markdown("")

# Feature cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 💬 ConvoAI")
    st.write("Natural conversational chatbot powered by Llama 3.1 via Groq.")
with col2:
    st.markdown("### 📚 RAG Chatbot")
    st.write("Upload PDFs and ask questions — answers grounded in your documents.")
with col3:
    st.markdown("### 🌐 Web RAG")
    st.write("Enter any URL and chat about its content instantly.")

st.divider()

st.markdown(
    """
    <div class="social-links">
        <a href="https://www.linkedin.com/in/kapilan-p-c-361188280/" target="_blank">LinkedIn</a>
        <a href="https://github.com/Kapilan0803" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
