# 🎬 RAG Video/Audio Chatbot POC

A **Retrieval-Augmented Generation (RAG)** chatbot that uses **video and audio files** as its knowledge base — built entirely with **free, local-first tools**.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🎥 **Media Support** | MP4, AVI, MKV, MOV, WEBM, MP3, WAV, M4A, OGG, FLAC |
| 📝 **Transcription** | OpenAI Whisper (local, no API needed) |
| 🧠 **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| 🗄️ **Vector DB** | ChromaDB (persists across restarts) |
| 🤖 **LLM** | OpenRouter free tier (LLaMA 3.1 8B) |
| 💬 **UI** | Streamlit with dark glassmorphism design |
| 📍 **Source Attribution** | Shows which file & timestamp answered the question |

---

## 🛠️ Prerequisites

### 1. Python 3.10+
```bash
python --version   # should be 3.10 or higher
```

### 2. FFmpeg (required for audio/video extraction)

**Windows:**
```powershell
# Option A — via winget
winget install ffmpeg

# Option B — via Chocolatey
choco install ffmpeg

# Option C — manual: download from https://ffmpeg.org/download.html
# and add the bin/ folder to your PATH
```

Verify: `ffmpeg -version`

### 3. OpenRouter API Key (Free)
1. Go to [https://openrouter.ai](https://openrouter.ai)
2. Sign up — **no credit card required**
3. Click **"Keys"** → **"Create Key"**
4. Copy the key

---

## 🚀 Quick Start

```powershell
# 1. Clone / navigate to the project
cd c:\Users\SagarBhanushali\Downloads\rag_poc

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
Copy-Item .env.example .env
# Open .env and paste your OpenRouter key

# 5. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 📂 Project Structure

```
rag_poc/
├── app.py                      # 🖥️  Streamlit UI (main entry point)
├── config.py                   # ⚙️  All configuration knobs
├── requirements.txt            # 📦  Python dependencies
├── .env.example                # 🔑  API key template
├── .env                        # 🔑  Your actual keys (git-ignored)
│
├── ingestion/
│   ├── audio_extractor.py      # 🎵  Extract audio from video/audio files
│   ├── transcriber.py          # 📝  Whisper transcription
│   └── chunker.py              # ✂️   Chunk + embed + store in ChromaDB
│
├── retrieval/
│   ├── vector_store.py         # 🗄️   ChromaDB wrapper (query, list, delete)
│   └── rag_chain.py            # 🤖  Build prompt + call OpenRouter LLM
│
├── chroma_db/                  # 📊  ChromaDB data (auto-created)
└── temp_audio/                 # 🗑️   Temp WAV files (auto-cleaned)
```

---

## ⚙️ Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `"base"` | `tiny` (fastest) → `large` (most accurate) |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformer model |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `OPENROUTER_MODEL` | `llama-3.1-8b-instruct:free` | Change to any free model on OpenRouter |
| `LLM_MAX_TOKENS` | `1024` | Max tokens in LLM response |

---

## 💡 How It Works

```
Video/Audio File
      │
      ▼
[FFmpeg + pydub]         Extract audio → 16kHz mono WAV
      │
      ▼
[Whisper (local)]        Speech-to-text transcription with timestamps
      │
      ▼
[LangChain splitter]     Split transcript into 500-char overlapping chunks
      │
      ▼
[sentence-transformers]  Embed each chunk (local, all-MiniLM-L6-v2)
      │
      ▼
[ChromaDB]               Persist vectors locally

──────────── At query time ────────────

User Question
      │
      ▼
[sentence-transformers]  Embed question
      │
      ▼
[ChromaDB]               Find top-5 most similar chunks
      │
      ▼
[OpenRouter LLM]         Generate answer from context
      │
      ▼
Answer + Source Attribution (file name + timestamp)
```

---

## 🆓 Free Models Available on OpenRouter

Change `OPENROUTER_MODEL` in `config.py` to any of these:

| Model ID | Notes |
|---|---|
| `meta-llama/llama-3.1-8b-instruct:free` | ✅ Default — fast & capable |
| `mistralai/mistral-7b-instruct:free` | Great for structured answers |
| `google/gemma-3-1b-it:free` | Smallest / fastest |
| `deepseek/deepseek-r1:free` | Strong reasoning |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ffmpeg not found` | Install FFmpeg and ensure it's in your PATH |
| `OPENROUTER_API_KEY not set` | Create `.env` from `.env.example` and add your key |
| Slow transcription | Use `WHISPER_MODEL = "tiny"` in `config.py` |
| Out of memory | Use `WHISPER_MODEL = "tiny"` or `"base"` on CPU |
| ChromaDB errors | Delete the `chroma_db/` folder and re-ingest |

---

## 📝 License
MIT — free for personal and commercial use.
