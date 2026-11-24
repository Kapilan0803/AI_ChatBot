"""LLM layer for the AI Chatbot Suite (no Streamlit dependency).

Supports several providers so the app keeps working when one of them retires a
model or hits its free-tier limits:

* gemini  – Google Gemini API, free key (no card) from https://aistudio.google.com/apikey   [default]
* groq    – Groq API (free key from https://console.groq.com)
* ollama  – models running on your own computer (no key, no internet, https://ollama.com)
"""
import os
from typing import Generator, Optional

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# Model names change often on free tiers. If one stops working, change it in the
# sidebar or set GEMINI_MODEL / GROQ_MODEL / OLLAMA_MODEL in .env.
PROVIDERS = {
    "gemini": {
        "label": "Google Gemini (free)",
        "short": "Gemini",
        "env_keys": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "key_url": "https://aistudio.google.com/apikey",
        "models": ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.6-flash"],
    },
    "groq": {
        "label": "Groq",
        "short": "Groq",
        "env_keys": ["GROQ_API_KEY"],
        "key_url": "https://console.groq.com",
        "models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    },
    "ollama": {
        "label": "Ollama (local, no key)",
        "short": "Ollama",
        "env_keys": [],
        "key_url": "https://ollama.com",
        "models": ["llama3.2", "qwen2.5", "phi3"],
    },
}

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
if DEFAULT_PROVIDER not in PROVIDERS:
    DEFAULT_PROVIDER = "gemini"

MAX_HISTORY_MESSAGES = 40  # keep the last N messages per chat session

SYSTEM_PROMPT = (
    "You are a friendly, helpful, and conversational assistant named ConvoAI. "
    "You engage with users in a natural, human-like manner. Keep answers concise, "
    "engaging, and fast."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


# ───────────────────────── Provider settings helpers ─────────────────────────
def needs_key(provider: str) -> bool:
    return bool(PROVIDERS[provider]["env_keys"])


def default_model(provider: str) -> str:
    return os.getenv(f"{provider.upper()}_MODEL") or PROVIDERS[provider]["models"][0]


def get_api_key(provider: str, custom_key: Optional[str] = None) -> str:
    """Return a usable key ('' if missing or still a placeholder such as 'your_...')."""
    candidates = [custom_key] + [os.getenv(name) for name in PROVIDERS[provider]["env_keys"]]
    for candidate in candidates:
        key = (candidate or "").strip()
        if key and not key.lower().startswith("your_"):
            return key
    return ""


def is_ready(provider: str, custom_key: Optional[str] = None) -> bool:
    return (not needs_key(provider)) or bool(get_api_key(provider, custom_key))


def key_missing_message(provider: str) -> str:
    cfg = PROVIDERS[provider]
    names = " or ".join(f"`{n}`" for n in cfg["env_keys"])
    return (
        f"⚠️ Please enter your **{cfg['short']} API key** in the sidebar "
        f"(free key: {cfg['key_url']}) or set {names} in `.env`."
    )


def make_llm(provider: str, api_key: str = "", model: Optional[str] = None, temperature: float = 0.7):
    """Create the chat model for a provider (imports are lazy, so only the one you use is needed)."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose one of: {', '.join(PROVIDERS)}")
    model = model or default_model(provider)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Temperature is left at Google's default (recommended for Gemini 3 models).
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, max_retries=2)

    if provider == "groq":
        from langchain_groq import ChatGroq

        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        return ChatGroq(api_key=api_key, model=model, temperature=temperature, **extra)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    raise ValueError(f"Unknown provider: {provider}")


def friendly_error(exc: Exception, provider: str) -> str:
    """Turn raw provider exceptions into a message the user can act on."""
    detail = f"{type(exc).__name__}: {exc}"
    low = detail.lower()
    label = PROVIDERS.get(provider, {}).get("label", provider)

    if provider == "ollama":
        if "not found" in low or "pull" in low:
            hint = "That Ollama model is not downloaded. Run `ollama pull <model>` (for example `ollama pull llama3.2`)."
        elif any(s in low for s in ("connect", "refused", "10061", "timed out", "unreachable")):
            hint = "Cannot reach Ollama. Install it from https://ollama.com, start it, then run `ollama pull llama3.2`."
        else:
            hint = ""
    elif any(s in low for s in ("429", "resource_exhausted", "rate limit", "rate_limit", "quota", "too many requests")):
        hint = f"The free-tier limit for {label} was reached. Wait a minute and try again, or pick another model/provider in the sidebar."
    elif any(s in low for s in ("401", "403", "api key", "api_key", "unauthorized", "permission_denied", "authentication")):
        hint = f"{label} rejected the API key. Check that you pasted the whole key for this provider."
    elif any(s in low for s in ("404", "model_not_found", "not found", "decommissioned", "no longer supported", "deprecated")):
        hint = f"The selected model is not available on {label} (retired or renamed). Pick another model in the sidebar."
    else:
        hint = ""

    return f"{hint}\n\nDetails: {detail[:400]}" if hint else detail[:600]


# ───────────────────────── Conversation with memory ─────────────────────────
_stores: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _stores:
        _stores[session_id] = InMemoryChatMessageHistory()
    return _stores[session_id]


def clear_history(session_id: str) -> None:
    """Forget the conversation for one session."""
    _stores.pop(session_id, None)


def chat_stream(
    user_input: str,
    session_id: str = "default",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    """Stream the reply token by token, remembering the conversation per session_id."""
    provider = provider or DEFAULT_PROVIDER
    if not is_ready(provider, api_key):
        raise ValueError(key_missing_message(provider))

    chain = (
        prompt
        | make_llm(provider, get_api_key(provider, api_key), model, temperature=0.7)
        | StrOutputParser()
    )
    with_history = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    yield from with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    # Keep memory bounded (runs once the stream has finished and been saved).
    history = _get_session_history(session_id)
    if len(history.messages) > MAX_HISTORY_MESSAGES:
        trimmed = history.messages[-MAX_HISTORY_MESSAGES:]
        if len(trimmed) % 2:  # keep human/AI pairs aligned
            trimmed = trimmed[1:]
        history.messages = trimmed


def chat_call(
    user_input: str,
    session_id: str = "default",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Non-streaming convenience wrapper."""
    return "".join(chat_stream(user_input, session_id, provider, api_key, model))


if __name__ == "__main__":
    print(f"Provider: {DEFAULT_PROVIDER} (change with LLM_PROVIDER in .env). Type 'quit' to exit.")
    while True:
        human_input = input("You: ")
        if human_input.lower() in ("quit", "exit"):
            break
        print("Bot: ", end="")
        try:
            for token in chat_stream(human_input):
                print(token, end="", flush=True)
        except Exception as e:  # noqa: BLE001
            print(friendly_error(e, DEFAULT_PROVIDER), end="")
        print()
