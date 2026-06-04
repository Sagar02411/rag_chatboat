"""
Central configuration for the RAG Video/Audio Chatbot POC.
All tunable parameters are here — no need to hunt through source files.
"""

import os
import ssl

# ---------------------------------------------------------------------------
# Corporate proxy / SSL fix
# Disables SSL certificate verification for HuggingFace Hub downloads.
# This is required when a corporate proxy uses self-signed certificates.
# ---------------------------------------------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context

os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("HTTPX_SSL_VERIFY", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ---------------------------------------------------------------------------
# FFmpeg PATH fix
# winget installs FFmpeg to a long path that isn't picked up automatically
# by child processes. We inject it here so shutil.which and subprocess both
# find ffmpeg.exe without requiring a shell restart.
# ---------------------------------------------------------------------------
_FFMPEG_BIN = (
    r"C:\Users\SagarBhanushali\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1.1-full_build\bin"
)
if _FFMPEG_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

# Patch httpx to skip SSL verification (used by huggingface_hub >= 0.23)
# Guard flag is stored ON the class itself so it survives importlib.reload(config).
try:
    import httpx
    if not getattr(httpx.Client, "_ssl_patch_applied", False):
        _orig_client_init = httpx.Client.__init__
        def _patched_client_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _orig_client_init(self, *args, **kwargs)
        httpx.Client.__init__ = _patched_client_init
        httpx.Client._ssl_patch_applied = True

    if not getattr(httpx.AsyncClient, "_ssl_patch_applied", False):
        _orig_async_init = httpx.AsyncClient.__init__
        def _patched_async_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _orig_async_init(self, *args, **kwargs)
        httpx.AsyncClient.__init__ = _patched_async_init
        httpx.AsyncClient._ssl_patch_applied = True
except Exception:
    pass  # httpx not installed — safe to ignore

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Transcription (Whisper — runs locally, no API key needed)
# ---------------------------------------------------------------------------
# Options: "tiny", "base", "small", "medium", "large"
# "base"  ~74 MB  — fast, good enough for POC
# "small" ~244 MB — better accuracy, still CPU-friendly
WHISPER_MODEL = "base"

# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Embeddings (sentence-transformers — runs locally, no API key needed)
# ---------------------------------------------------------------------------
# After running `python download_model.py` once, the model lives in ./models/
# and is loaded fully offline — no HuggingFace network calls at runtime.
EMBEDDING_MODEL = "./models/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Vector Store (ChromaDB — local persistent storage)
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION  = "video_audio_kb"

# Number of similar chunks to retrieve per query
TOP_K_RESULTS = 5

# ---------------------------------------------------------------------------
# LLM — OpenRouter (free tier, OpenAI-compatible API)
# Sign up free at https://openrouter.ai — no credit card required.
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models available on OpenRouter (use openrouter/free to auto-pick):
# The smart router below always picks a currently-available free model.
# Fallback options if you want a specific model:
#   "deepseek/deepseek-r1:free"           — strong reasoning
#   "meta-llama/llama-3.2-3b-instruct:free"
#   "poolside/laguna-xs.2:free"             — confirmed free, 262K ctx, works
#   "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free" — free but reasoning-only
#   "google/gemma-2-9b-it:free"             — great marketing tone (may be offline)
#   "mistralai/mistral-7b-instruct:free"    — fast & concise (may be offline)
OPENROUTER_MODEL = "poolside/laguna-xs.2:free"  # confirmed free & returns content


# Max tokens for LLM response
LLM_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Temp directory for extracted audio files
# ---------------------------------------------------------------------------
TEMP_AUDIO_DIR = "./temp_audio"

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
SUPPORTED_EXTS       = SUPPORTED_VIDEO_EXTS | SUPPORTED_AUDIO_EXTS
