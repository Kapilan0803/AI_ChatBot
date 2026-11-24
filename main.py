import streamlit as st

from utils import LOGO_PATH, render_sidebar_settings

# --- PAGE CONFIG (must be the first Streamlit call) ---
st.set_page_config(page_title="AI Chatbot Suite", page_icon="🤖", layout="centered")

# --- SHARED ON ALL PAGES ---
if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH))

# --- PAGE SETUP ---
pages = {
    " ": [
        st.Page(
            "home_page.py",
            title="HOME",
            icon=":material/home:",
            default=True,
        )
    ],
    "Tools": [
        st.Page(
            "Chatbot.py",
            title="Conversation_chat",
            icon=":material/forum:",
        ),
        st.Page(
            "RAG_CODE.py",
            title="RAG MODEL",
            icon=":material/picture_as_pdf:",
        ),
        st.Page(
            "WEB_BASED.py",
            title="WEB-RAG MODEL",
            icon=":material/public:",
        ),
    ],
}

# --- NAVIGATION SETUP [WITH SECTIONS] ---
pg = st.navigation(pages)

# API key + model selector, shared by every page
render_sidebar_settings()

# --- RUN NAVIGATION ---
pg.run()
