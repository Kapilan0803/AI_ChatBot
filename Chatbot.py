import uuid

import streamlit as st

from llm import PROVIDERS, chat_stream, clear_history
from utils import (
    explain_error,
    get_api_key,
    get_model,
    get_provider,
    key_missing_message,
    provider_ready,
)

WELCOME = {"role": "assistant", "content": "Hello buddy! How can I help you today?"}

st.title("💬🤖 ConvoAI")
st.caption(f"🚀 Fast conversation powered by {PROVIDERS[get_provider()]['label']} · {get_model()}")

# ── Session state ──
# Each browser session gets its own id, so conversations never mix between users.
if "chat_session_id" not in st.session_state:
    st.session_state["chat_session_id"] = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state["messages"] = [dict(WELCOME)]

with st.sidebar:
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        # Clear the model's memory too, not just what is shown on screen.
        clear_history(st.session_state["chat_session_id"])
        st.session_state["messages"] = [dict(WELCOME)]
        st.rerun()

# ── Display chat history ──
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ── User input ──
if prompt_text := st.chat_input("Ask me anything..."):
    st.chat_message("user").write(prompt_text)

    with st.chat_message("assistant"):
        if not provider_ready():
            st.error(key_missing_message())
        else:
            try:
                reply = st.write_stream(
                    chat_stream(
                        prompt_text,
                        session_id=st.session_state["chat_session_id"],
                        provider=get_provider(),
                        api_key=get_api_key(),
                        model=get_model(),
                    )
                )
            except Exception as e:
                st.error(explain_error(e))
            else:
                st.session_state["messages"].append({"role": "user", "content": prompt_text})
                st.session_state["messages"].append({"role": "assistant", "content": reply})
