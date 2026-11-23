from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq

import os
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

# ── Chat prompt with system message and history ──
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly, helpful, and conversational assistant named ConvoAI. "
     "You engage with users in a natural, human-like manner. Keep answers concise, "
     "engaging, and fast."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# ── In-memory session store (keyed per Streamlit session) ──
_stores: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _stores:
        _stores[session_id] = ChatMessageHistory()
    return _stores[session_id]


def get_groq_api_key(custom_key: str = None) -> str:
    key = custom_key or os.getenv("GROQ_API_KEY", "")
    if key and key != "your_api_key":
        return key
    return ""


def Llama_stream(user_input: str, session_id: str = "default", api_key: str = None) -> Generator[str, None, None]:
    """Stream response tokens instantly from Groq at 800+ tokens/sec."""
    key = get_groq_api_key(api_key)
    if not key:
        raise ValueError("Missing Groq API Key. Please provide a valid Groq API Key in .env or the sidebar.")

    llm = ChatGroq(
        groq_api_key=key,
        model_name="llama-3.1-8b-instant",
        streaming=True,
        temperature=0.7,
    )

    chain = prompt | llm | StrOutputParser()

    with_history = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    for chunk in with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk


def Llama_call(user_input: str, session_id: str = "default", api_key: str = None) -> str:
    """Non-streaming fallback."""
    return "".join(Llama_stream(user_input, session_id, api_key))


if __name__ == "__main__":
    while True:
        human_input = input("You: ")
        if human_input.lower() in ("quit", "exit"):
            break
        print("Bot: ", end="")
        for token in Llama_stream(human_input):
            print(token, end="", flush=True)
        print()
