"""
audio_extractor.py
------------------
Extracts audio from video/audio files and returns a path suitable for
Whisper transcription.

FFmpeg is required for:
  - All VIDEO files (mp4, avi, mkv, mov, webm)
  - Non-WAV AUDIO files (mp3, m4a, ogg, flac)

WAV files are passed through directly — no FFmpeg needed.
"""

import os
import shutil
import logging
import subprocess
from pathlib import Path

from config import TEMP_AUDIO_DIR, SUPPORTED_VIDEO_EXTS, SUPPORTED_AUDIO_EXTS

logger = logging.getLogger(__name__)

FFMPEG_INSTALL_MSG = (
    "FFmpeg is not installed or not found in PATH.\n\n"
    "Install it with one of these commands:\n"
    "  winget install ffmpeg\n"
    "  choco install ffmpeg\n\n"
    "Or download manually from https://ffmpeg.org/download.html\n"
    "and add the 'bin' folder to your system PATH.\n\n"
    "After installing, restart this app."
)


def _check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio(file_path: str) -> str:
    """
    Given a path to a video or audio file, returns a path to a WAV file
    suitable for Whisper transcription.

    For WAV files: returned as-is (no FFmpeg needed).
    For all other formats: FFmpeg is used to decode and convert to 16kHz mono WAV.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the source media file.

    Returns
    -------
    str
        Path to the WAV file ready for Whisper.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the source file does not exist.
    RuntimeError
        If FFmpeg is required but not installed.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    all_exts = SUPPORTED_VIDEO_EXTS | SUPPORTED_AUDIO_EXTS
    if ext not in all_exts:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {sorted(all_exts)}"
        )

    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

    # ── WAV shortcut: no conversion needed ──────────────────────────────────
    if ext == ".wav":
        logger.info(f"WAV file — using directly: {file_path}")
        return str(file_path)

    # ── All other formats require FFmpeg ────────────────────────────────────
    if not _check_ffmpeg():
        raise RuntimeError(FFMPEG_INSTALL_MSG)

    out_wav = os.path.join(TEMP_AUDIO_DIR, path.stem + "_extracted.wav")
    logger.info(f"Converting '{path.name}' → '{out_wav}' via FFmpeg")

    # Use ffmpeg directly for reliable cross-format support
    cmd = [
        "ffmpeg",
        "-y",                   # overwrite output
        "-i", str(path),        # input file
        "-vn",                  # no video stream
        "-acodec", "pcm_s16le", # 16-bit WAV
        "-ar", "16000",         # 16 kHz sample rate (ideal for Whisper)
        "-ac", "1",             # mono
        out_wav,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,         # 10 min max
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg failed:\n{err}")
    except FileNotFoundError:
        raise RuntimeError(FFMPEG_INSTALL_MSG)
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out after 10 minutes — file may be too large.")

    logger.info(f"Audio extracted successfully: {out_wav}")
    return out_wav


def cleanup_temp_audio(wav_path: str) -> None:
    """Remove a temporary WAV file after transcription is complete."""
    try:
        # Don't delete the original if it was a WAV passed through directly
        if wav_path and "_extracted.wav" in wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
            logger.debug(f"Deleted temp file: {wav_path}")
    except Exception as e:
        logger.warning(f"Could not delete temp file '{wav_path}': {e}")
