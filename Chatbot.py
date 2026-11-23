import streamlit as st
import os
from dotenv import load_dotenv
from Llama import Llama_stream

load_dotenv()

# Streamlit app setup
st.title("💬🤖 ConvoAI")
st.caption("🚀 Lightning-fast conversation powered by Llama 3.1 & Groq")
st.logo("Chatbots-removebg-preview.png")

# Sidebar API Key input for instant configuration
env_key = os.getenv("GROQ_API_KEY", "")
if env_key == "your_api_key":
    env_key = ""

with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key_input = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Get your free API key at https://console.groq.com",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello buddy! How can I help you today?"}]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello buddy! How can I help you today?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt_text := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.chat_message("user").write(prompt_text)

    active_key = api_key_input or os.getenv("GROQ_API_KEY", "")
    if not active_key or active_key == "your_api_key":
        with st.chat_message("assistant"):
            st.error("⚠️ Please enter a valid **Groq API Key** in the sidebar or in `.env` to chat.")
    else:
        with st.chat_message("assistant"):
            try:
                # Real-time token streaming - renders immediately with zero lag!
                stream_generator = Llama_stream(prompt_text, api_key=active_key)
                full_response = st.write_stream(stream_generator)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {e}")
