"""
app.py — RAG Video/Audio Chatbot POC
=====================================
Run with:  streamlit run app.py
"""

import logging
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Configure logging ──────────────────────────────────────────────────────
# Guard with a flag on the root logger so this only runs ONCE per process,
# not on every Streamlit rerun (which would cause duplicate handlers).
_root = logging.getLogger()
if not getattr(_root, "_rag_logging_configured", False):
    _root.setLevel(logging.INFO)
    _root.handlers.clear()  # remove Streamlit's default handlers

    _log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    _file_handler = logging.FileHandler("tmp.log", mode="a", encoding="utf-8")
    _file_handler.setFormatter(_log_fmt)
    _root.addHandler(_file_handler)

    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_log_fmt)
    _root.addHandler(_console_handler)

    _root._rag_logging_configured = True  # type: ignore[attr-defined]

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="🎬 RAG Media Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after page config) ────────────────────────────────────────────
from config import SUPPORTED_EXTS, OPENROUTER_API_KEY
from ingestion.audio_extractor import extract_audio, cleanup_temp_audio
from ingestion.transcriber import transcribe
from ingestion.chunker import ingest_transcript
from retrieval.vector_store import query, list_sources, delete_source, chunk_count
from retrieval.rag_chain import answer, format_sources, LANGUAGES

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global fonts & background ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
        min-height: 100vh;
    }

    /* Make all Streamlit inner containers transparent so dark bg shows through */
    .stMain, .stMainBlockContainer, .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    section.main > div {
        background: transparent !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #f0f2f8;
        border-right: 2px solid #d0d4e8;
    }

    /* Sidebar text — dark so it's readable on the light background */
    [data-testid="stSidebar"] * {
        color: #1a1a2e !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #1a1a2e !important;
    }

    /* ── Chat bubbles ── */
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #ffffff !important;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0 8px auto;
        max-width: 78%;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .chat-user, .chat-user * { color: #ffffff !important; }

    .chat-bot {
        background: #1e1e3c !important;
        border: 1px solid rgba(167,139,250,0.5) !important;
        color: #f0f0ff !important;
        border-radius: 18px 18px 18px 4px;
        padding: 16px 20px;
        margin: 8px 0;
        max-width: 85%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        font-size: 0.95rem;
        line-height: 1.7;
    }
    /* Force ALL text inside bot bubble to be white — overrides Streamlit theme */
    .chat-bot,
    .chat-bot *,
    .chat-bot p,
    .chat-bot li,
    .chat-bot span,
    .chat-bot div,
    .chat-bot h1,
    .chat-bot h2,
    .chat-bot h3,
    .chat-bot h4,
    .chat-bot ul,
    .chat-bot ol { color: #f0f0ff !important; }
    .chat-bot strong, .chat-bot b { color: #c4b5fd !important; }
    .chat-bot em, .chat-bot i  { color: #a5f3fc !important; }
    .chat-bot code { color: #a5f3fc !important; background: rgba(0,0,0,0.35) !important; padding: 2px 6px; border-radius: 4px; }
    .chat-bot a { color: #818cf8 !important; }

    .source-badge {
        background: rgba(80, 60, 160, 0.5);
        border: 1px solid rgba(167,139,250,0.5);
        border-radius: 10px;
        padding: 8px 14px;
        margin-top: 6px;
        font-size: 0.82rem;
        color: #ddd6fe !important;
        max-width: 85%;
        word-break: break-word;
    }
    .source-badge * { color: #ddd6fe !important; }

    /* ── File cards ── */
    .file-card {
        background: rgba(102,126,234,0.12);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: #1a1a2e;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        border: 1px solid #d0d4e8;
        box-shadow: 0 2px 8px rgba(102,126,234,0.12);
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #5b4fcf; }
    .metric-label { font-size: 0.78rem; color: #555577; margin-top: 2px; }

    /* ── Input box ── */
    .stChatInput > div { border-radius: 16px !important; }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }

    /* ── Progress bar ── */
    .stProgress > div > div { background: linear-gradient(90deg, #667eea, #a78bfa) !important; }

    /* ── Hero ── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
    }
    .hero-sub {
        text-align: center;
        color: #8b8fa8;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #4a4d6a;
    }
    .empty-state .icon { font-size: 4rem; margin-bottom: 12px; }
    .empty-state p { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": "user"|"assistant", "content": str, "sources": str}

if "ingested_files" not in st.session_state:
    # Pre-populate from existing ChromaDB data
    try:
        st.session_state.ingested_files = list_sources()
    except Exception:
        st.session_state.ingested_files = []

if "transcripts" not in st.session_state:
    st.session_state.transcripts = {}  # {filename: full_text}

# ── Ingestion function (defined here so it's available when sidebar runs) ────
def _ingest_file(uploaded_file):
    """Handle the full ingestion pipeline for an uploaded file."""
    fname = uploaded_file.name
    progress_bar = st.sidebar.progress(0, text=f"Processing {fname}…")
    status_text  = st.sidebar.empty()

    try:
        # Save to temp file
        status_text.info("💾 Saving uploaded file…")
        progress_bar.progress(10, text="Saving…")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(fname).suffix,
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Extract audio
        status_text.info("🎵 Extracting audio…")
        progress_bar.progress(25, text="Extracting audio…")
        wav_path = extract_audio(tmp_path)

        # Transcribe
        status_text.info("📝 Transcribing with Whisper… (may take a while for large files)")
        progress_bar.progress(45, text="Transcribing…")
        transcript = transcribe(wav_path)

        if not transcript["full_text"].strip():
            st.sidebar.error(f"❌ No speech detected in '{fname}'.")
            return

        # Chunk + embed + store
        status_text.info("🔢 Embedding & storing in ChromaDB…")
        progress_bar.progress(75, text="Storing embeddings…")
        n_chunks = ingest_transcript(
            full_text=transcript["full_text"],
            source_name=fname,
            segments=transcript["segments"],
        )

        # Cleanup
        cleanup_temp_audio(wav_path)
        os.remove(tmp_path)

        progress_bar.progress(100, text="Done!")
        status_text.success(
            f"✅ '{fname}' ingested!\n"
            f"- Language: {transcript['language']}\n"
            f"- Chunks stored: {n_chunks}\n"
            f"- Segments: {len(transcript['segments'])}"
        )

        if fname not in st.session_state.ingested_files:
            st.session_state.ingested_files.append(fname)

        # Store transcript text for sidebar display
        st.session_state.transcripts[fname] = transcript["full_text"]

        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        err_str = str(e)
        ffmpeg_hint = (
            "\n\n💡 **FFmpeg not found.** Install it with:\n"
            "```\nwinget install ffmpeg\n```\n"
            "Then restart your terminal and re-run `streamlit run app.py`.\n"
            "_(WAV files work without FFmpeg.)_"
            if ("WinError 2" in err_str or "ffmpeg" in err_str.lower() or "FFmpeg" in err_str)
            else ""
        )
        status_text.error(f"❌ Error processing '{fname}':\n\n{err_str}{ffmpeg_hint}")
        logging.exception(f"Ingestion failed for '{fname}'")


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Marketing Chatbot")
    st.markdown("---")

    # ── API key check ──────────────────────────────────────────────────────
    if not OPENROUTER_API_KEY:
        st.error(
            "⚠️ **OpenRouter API key missing!**\n\n"
            "1. Get a free key at [openrouter.ai](https://openrouter.ai)\n"
            "2. Copy `.env.example` → `.env`\n"
            "3. Paste your key and restart the app.",
            icon="🔑",
        )

    # ── FFmpeg check ───────────────────────────────────────────────────────
    import shutil as _shutil
    _ffmpeg_ok = _shutil.which("ffmpeg") is not None
    if not _ffmpeg_ok:
        st.warning(
            "⚠️ **FFmpeg not found!**\n\n"
            "FFmpeg is required to process **MP3, MP4, M4A** and other formats.\n\n"
            "**Install (run in a new terminal):**\n"
            "```\nwinget install ffmpeg\n```\n"
            "Then close & reopen your terminal and restart the app.\n\n"
            "💡 *WAV files work without FFmpeg.*",
        )
    else:
        st.success("FFmpeg detected")

    # ── Model selector ─────────────────────────────────────────────────────
    st.markdown("### 🤖 LLM Model")
    FREE_MODELS = {
        "Poolside Laguna XS.2 (free, recommended)": "poolside/laguna-xs.2:free",
        "Poolside Laguna M.1 (free)": "poolside/laguna-m.1:free",
        "NVIDIA Nemotron 30B (free, reasoning)": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "Llama 3.1 8B (free, may be offline)": "meta-llama/llama-3.1-8b-instruct:free",
        "Gemma 2 9B (free, may be offline)": "google/gemma-2-9b-it:free",
        "DeepSeek R1 (free, may be offline)": "deepseek/deepseek-r1:free",
        "Mistral 7B (free, may be offline)": "mistralai/mistral-7b-instruct:free",
    }
    _model_labels = list(FREE_MODELS.keys())
    if "selected_model_label" not in st.session_state:
        st.session_state.selected_model_label = _model_labels[0]

    chosen_label = st.selectbox(
        "Choose LLM",
        options=_model_labels,
        index=_model_labels.index(st.session_state.selected_model_label),
        key="model_selectbox",
        help="Switch models instantly — no restart needed.",
    )
    st.session_state.selected_model_label = chosen_label
    st.session_state.selected_model = FREE_MODELS[chosen_label]
    st.caption(f"`{st.session_state.selected_model}`")

    # ── Response language selector ─────────────────────────────────────
    st.markdown("### 🌐 Response Language")
    _lang_labels = list(LANGUAGES.keys())
    if "selected_lang_label" not in st.session_state:
        st.session_state.selected_lang_label = _lang_labels[0]  # Auto-detect

    chosen_lang = st.selectbox(
        "Respond in",
        options=_lang_labels,
        index=_lang_labels.index(st.session_state.selected_lang_label),
        key="lang_selectbox",
        help="Auto-detect mirrors the language of your question. Force a language for consistent output.",
    )
    st.session_state.selected_lang_label = chosen_lang
    st.session_state.selected_response_language = LANGUAGES[chosen_lang]  # None or e.g. "Japanese"
    if st.session_state.selected_response_language:
        st.caption(f"Forcing output in: **{st.session_state.selected_response_language}**")
    else:
        st.caption("Auto-detecting from your question.")
    st.markdown("---")

    # ── Stats ──────────────────────────────────────────────────────────────
    try:
        total_chunks = chunk_count()
        total_files  = len(list_sources())
    except Exception:
        total_chunks = 0
        total_files  = 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_files}</div>'
            f'<div class="metric-label">Files</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{total_chunks}</div>'
            f'<div class="metric-label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Media")
    ext_list = ", ".join(sorted(SUPPORTED_EXTS))
    uploaded = st.file_uploader(
        f"Supported: {ext_list}",
        type=[e.lstrip(".") for e in SUPPORTED_EXTS],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if uploaded:
        for uf in uploaded:
            file_ext = Path(uf.name).suffix.lower()
            if file_ext not in SUPPORTED_EXTS:
                st.warning(f"Skipping unsupported file: {uf.name}")
                continue

            already_ingested = uf.name in st.session_state.ingested_files
            btn_label = f"{'✅ Re-ingest' if already_ingested else '⚡ Ingest'}: {uf.name}"

            if st.button(btn_label, key=f"ingest_{uf.name}", use_container_width=True):
                _ingest_file(uf)

    st.markdown("---")

    # ── Knowledge base file list ───────────────────────────────────────────
    st.markdown("### 🗄️ Knowledge Base")
    kb_sources = list_sources()

    if not kb_sources:
        st.caption("_No files ingested yet. Upload a video or audio file above._")
    else:
        for src in kb_sources:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                icon = "🎥" if any(src.lower().endswith(e) for e in SUPPORTED_EXTS - {".mp3",".wav",".m4a",".ogg",".flac"}) else "🎵"
                st.markdown(
                    f'<div class="file-card">{icon} {src}</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_{src}", help=f"Remove {src} from knowledge base"):
                    removed = delete_source(src)
                    if src in st.session_state.ingested_files:
                        st.session_state.ingested_files.remove(src)
                    st.success(f"Removed {removed} chunks for '{src}'")
                    st.rerun()

    st.markdown("---")

    # ── Transcripts ────────────────────────────────────────────────────────
    st.markdown("### 📄 Transcripts")
    if not st.session_state.transcripts:
        st.caption(
            "_Transcripts appear here after you ingest a file. "
            "Re-ingest an existing file to load its transcript._"
        )
    else:
        for filename, text in st.session_state.transcripts.items():
            word_count = len(text.split())
            char_count = len(text)
            with st.expander(f"📎 {filename}  ({word_count:,} words)", expanded=False):
                st.caption(f"{char_count:,} characters · {word_count:,} words")
                st.text_area(
                    label="Full transcript",
                    value=text,
                    height=300,
                    key=f"transcript_{filename}",
                    label_visibility="collapsed",
                )

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">🎬 RAG Media Chatbot</div>'
    '<div class="hero-sub">Ask anything about your uploaded videos & audio files</div>',
    unsafe_allow_html=True,
)

# ── Display chat history ───────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">🎙️</div>
                <p><strong>No conversations yet.</strong><br>
                Upload a video or audio file in the sidebar,<br>
                then ask a question below!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    st.markdown(
                        f'<div class="source-badge">{msg["sources"]}</div>',
                        unsafe_allow_html=True,
                    )

# ── Chat input ─────────────────────────────────────────────────────────────
user_question = st.chat_input(
    "Ask about your marketing content…",
    disabled=(not OPENROUTER_API_KEY),
)

if user_question:
    # Guard: need ingested content
    if chunk_count() == 0:
        st.warning("⚠️ Please ingest at least one video/audio file first (sidebar → Upload Media).")
    else:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Retrieve relevant chunks
        with st.spinner("🔍 Searching knowledge base…"):
            chunks = query(user_question)

        if not chunks:
            bot_reply = "I could not find any relevant information in the ingested media files."
            sources_md = ""
        else:
            # Stream the LLM answer
            answer_placeholder = st.empty()
            full_answer = ""

            with st.spinner("Generating answer..."):
                try:
                    token_stream = answer(
                        question=user_question,
                        context_chunks=chunks,
                        stream=True,
                        model=st.session_state.get("selected_model"),
                        response_language=st.session_state.get("selected_response_language"),
                    )
                    for token in token_stream:
                        full_answer += token
                        answer_placeholder.markdown(
                            f'<div class="chat-bot">🤖 {full_answer}▌</div>',
                            unsafe_allow_html=True,
                        )
                    answer_placeholder.empty()
                except Exception as e:
                    full_answer = f"⚠️ LLM error: {e}"

            bot_reply   = full_answer
            sources_md  = format_sources(chunks)

        # Save assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": bot_reply, "sources": sources_md}
        )
        st.rerun()
