"""
transcriber.py
--------------
Wraps OpenAI Whisper for local, offline speech-to-text transcription.
The model is downloaded once (~74 MB for "base") and cached by Whisper.
"""

import logging
from typing import TypedDict

import whisper

from config import WHISPER_MODEL

logger = logging.getLogger(__name__)

# Module-level cache so the model is only loaded once per session
_whisper_model = None


def _get_model():
    """Lazy-load and cache the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"Loading Whisper model '{WHISPER_MODEL}' (first run downloads it)…")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        logger.info("Whisper model loaded.")
    return _whisper_model


class TranscriptSegment(TypedDict):
    start: float   # seconds
    end: float     # seconds
    text: str


class TranscriptResult(TypedDict):
    full_text: str
    segments: list[TranscriptSegment]
    language: str


def transcribe(audio_path: str) -> TranscriptResult:
    """
    Transcribe a WAV (or any audio) file using local Whisper.

    Parameters
    ----------
    audio_path : str
        Path to the audio file to transcribe.

    Returns
    -------
    TranscriptResult
        Dictionary with:
        - ``full_text``: complete transcript as a single string
        - ``segments``: list of dicts with ``start``, ``end``, ``text``
        - ``language``: detected language code (e.g. "en")
    """
    model = _get_model()

    logger.info(f"Transcribing: {audio_path}")
    result = model.transcribe(
        audio_path,
        fp16=False,          # fp16=False → safe for CPU-only machines
        verbose=False,
    )

    segments: list[TranscriptSegment] = [
        {
            "start": seg["start"],
            "end":   seg["end"],
            "text":  seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]

    full_text: str = result.get("text", "").strip()
    logger.info(f"Full text: {full_text}")
    language: str  = result.get("language", "unknown")

    logger.info(
        f"Transcription done — {len(full_text)} chars, "
        f"{len(segments)} segments, language='{language}'"
    )

    return TranscriptResult(
        full_text=full_text,
        segments=segments,
        language=language,
    )
