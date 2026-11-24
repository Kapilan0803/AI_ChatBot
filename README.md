# AI Chatbot Suite

A Streamlit app with three chatbots, built with LangChain. You choose the AI provider in the sidebar.

| Page | What it does |
|------|--------------|
| 💬 **ConvoAI** | General conversational chatbot with per-session memory |
| 📚 **RAG Model** | Upload a PDF or text file and ask questions grounded in that document |
| 🌐 **Web-RAG Model** | Paste a URL and chat about the content of that web page |

---

## AI providers

| Provider | Needs | Default model |
|----------|-------|---------------|
| **Google Gemini** (default) | Free API key, no credit card: <https://aistudio.google.com/apikey> | `gemini-3.5-flash-lite` |
| **Groq** | Free API key: <https://console.groq.com> | `openai/gpt-oss-20b` |
| **Ollama** (runs on your PC) | Install <https://ollama.com>, then `ollama pull llama3.2` – no key, no internet | `llama3.2` |

Free-tier model names change often (Groq retired `llama-3.1-8b-instant` on 16 Aug 2026). If a model stops working,
pick another one in the sidebar, or set `GEMINI_MODEL` / `GROQ_MODEL` / `OLLAMA_MODEL` in `.env`.
The Gemini free tier has daily/per-minute limits, and Google may use free-tier prompts to improve its products,
so avoid uploading confidential documents when using it (use Ollama for private files).

---

## Technology Stack
- **Python 3.10+**
- [Streamlit](https://streamlit.io/) – web interface and multipage navigation
- [LangChain](https://github.com/langchain-ai/langchain) – prompts, chains, loaders, text splitting
- Gemini / Groq / Ollama through `langchain-google-genai`, `langchain-groq`, `langchain-ollama`
- [Sentence-Transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` – local embeddings (no key needed)
- [FAISS](https://github.com/facebookresearch/faiss) – in-memory vector search (falls back to LangChain's built-in in-memory store if FAISS is not installed)

---

## Setup

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. add your key (Gemini is the default)
copy .env.example .env          # Windows  (cp .env.example .env on macOS / Linux)
# then edit .env and set GOOGLE_API_KEY=...

# 4. run
streamlit run main.py

---

## File Structure
```
ai-chatbot-suite/
├── main.py              # Entry point: page config, navigation, shared sidebar (provider, key, model)
├── home_page.py         # Home page
├── Chatbot.py           # Conversational chatbot page
├── RAG_CODE.py          # PDF / text RAG page
├── WEB_BASED.py         # Web-page RAG page
├── llm.py               # Providers (Gemini / Groq / Ollama), key handling, chat chain with memory
├── utils.py             # Shared helpers: settings, loaders, embeddings, vector index, RAG answering
├── requirements.txt
├── .env.example
└── Chatbots-removebg-preview.png   # logo (optional)
```

## Contributing
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description.

## License
Released under the MIT License (add a `LICENSE` file to the repository so GitHub shows it).

## Acknowledgments
- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [Google Gemini](https://ai.google.dev/), [Groq](https://groq.com/), [Ollama](https://ollama.com/)
