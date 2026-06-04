# local-pdf-rag

> Ask questions about any PDF — entirely on your own machine. No API keys, no data leaving your device.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-blue)
![Ollama](https://img.shields.io/badge/Ollama-local_LLM-black)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why Local-First RAG?

Most RAG tutorials send your documents to OpenAI or a cloud embedding service. That's fine for demos, but it's a problem when the documents contain anything sensitive — contracts, medical records, internal reports, student data.

This project runs the entire pipeline locally: extraction, embedding, retrieval, and generation. Your PDFs stay on your machine.

It's also deliberately minimal — no LangChain, no orchestration framework. Just the four core components of RAG, written directly, so you can see exactly what's happening at each step.

---

## How It Works

```
PDF upload
    │
    ▼
Text extraction (PyPDF2)
    │
    ▼
Chunking — fixed-size windows with overlap
    │
    ▼
Embedding (SentenceTransformers: all-MiniLM-L6-v2)
    │                                ↕ persisted to chunks.npy + chunks.index
    ▼
FAISS vector index
    
    
User asks a question
    │
    ▼
Query embedded → cosine similarity search
    │
    ▼
Top-K chunks retrieved
    │
    ▼
Chunks + question → Ollama prompt
    │
    ▼
Answer streamed back to Streamlit UI
```

No frameworks hiding the joins between steps. Each stage is a direct function call you can inspect and modify.

---

## Stack

| Component | Technology | Why |
|---|---|---|
| UI | Streamlit | Fast to run, easy to modify |
| PDF extraction | PyPDF2 | Lightweight, zero config |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Small model, runs on CPU |
| Vector search | FAISS (`faiss-cpu`) | Fast, local, no server required |
| LLM inference | Ollama | Any model you have pulled locally |

---

## Screenshot

![App screenshot](https://private-user-images.githubusercontent.com/12972773/376429244-911b4d39-6c16-46aa-8b7b-f528c534024f.png)

---

## Quickstart

### Prerequisites

- Python 3.x
- [Ollama](https://ollama.ai) running locally
- A model pulled: `ollama pull llama3.2` (or any chat model)

### Install and run

```bash
git clone https://github.com/nawzaysfinah/local-pdf-rag.git
cd local-pdf-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install streamlit PyPDF2 sentence-transformers faiss-cpu numpy

streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Using the App

1. **Upload a PDF** — drag and drop or use the file picker
2. The app extracts text, chunks it, embeds each chunk, and builds a FAISS index
3. **Ask a question** in the text input
4. The app retrieves the most relevant chunks and passes them to Ollama
5. The answer appears in the chat — grounded in your document

The index is cached per session so subsequent questions are fast.

---

## Troubleshooting

**Ollama not found** — Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.2`).

**`pull model manifest: file does not exist`** — The model name in the code doesn't match what you have. Check `ollama list` and update the model name in `app.py`.

**PDF not processing** — Ensure the PDF is not password-protected and is text-based (not a scanned image). For scanned PDFs, see [`pdf-knowledge-graph-pipeline`](https://github.com/nawzaysfinah/pdf-knowledge-graph-pipeline) which adds OCR.

**HuggingFace tokenizer warning** — Harmless. Suppressed by `os.environ["TOKENIZERS_PARALLELISM"] = "false"` in the script.

---

## When to Use This vs. Related Projects

| You want to... | Use |
|---|---|
| Understand RAG fundamentals with minimal abstraction | **this repo** |
| Build a full app with sessions, upload UI, streaming | [`build-llm-apps`](https://github.com/nawzaysfinah/build-llm-apps) |
| Extract structured knowledge (entities + relationships) from PDFs | [`pdf-knowledge-graph-pipeline`](https://github.com/nawzaysfinah/pdf-knowledge-graph-pipeline) |

---

*Built by [Syaz](https://syaz.super.site) — AI Lecturer @ ITE College West, Singapore*
