"""
rag_chain.py
------------
Builds the RAG prompt from retrieved context chunks and calls the
OpenRouter LLM (OpenAI-compatible API, free tier).
"""

import logging
from typing import Generator

from openai import OpenAI

import importlib
import config

logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    """Create a fresh OpenAI client each call — avoids 'client closed' on Streamlit reruns."""
    if not config.OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example → .env and add your free key from https://openrouter.ai"
        )
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an enthusiastic and knowledgeable marketing assistant representing the brand.
You answer questions based ONLY on the provided transcript context from marketing videos and audio files.

Your personality:
- Warm, professional and persuasive — like a top-tier sales consultant
- Highlight key benefits and unique selling points from the content
- Use clear, engaging language that resonates with customers
- Keep answers concise but impactful

Language rule (CRITICAL — follow this above all else):
- Detect the language the user wrote their question in.
- ALWAYS respond in that SAME language. If the question is in Japanese, answer in Japanese. If English, answer in English.
- If a specific response language is specified below, use that language regardless of the question language.

Rules:
- Base your answer STRICTLY on the provided context. Do NOT invent features, prices, or claims.
- If something is not covered in the context, say the equivalent of "That's a great question! For more details on that, I'd recommend speaking with our team directly." in the appropriate language.
- When relevant, mention which video/audio source the info comes from.
- End answers with a light call-to-action when appropriate.
- Format your response in clean, readable markdown — use bullet points for feature lists."""

LANGUAGES = {
    "Auto-detect (mirror question language)": None,
    "English": "English",
    "Japanese (日本語)": "Japanese",
    "Hindi (हिंدी)": "Hindi",
    "Chinese Simplified (简体中文)": "Simplified Chinese",
    "Spanish (Español)": "Spanish",
    "French (Français)": "French",
    "German (Deutsch)": "German",
    "Korean (한국어)": "Korean",
}


def _build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    """Assemble the user-facing prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_label = f"[{chunk['source']} @ {chunk['ts_label']}]" if chunk.get("ts_label") else f"[{chunk['source']}]"
        context_parts.append(f"--- Context {i} {source_label} ---\n{chunk['text']}")

    context_text = "\n\n".join(context_parts)

    return (
        f"Here is the relevant transcript context:\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer(
    question: str,
    context_chunks: list[dict],
    stream: bool = True,
    model: str | None = None,
    response_language: str | None = None,
) -> Generator[str, None, None] | str:
    """
    Generate an answer using the OpenRouter LLM.

    Parameters
    ----------
    question : str
        The user's question.
    context_chunks : list[dict]
        Retrieved chunks from vector_store.query().
    stream : bool
        If True, returns a generator that yields text tokens.
        If False, returns the full response string.
    model : str | None
        Override the model from config. If None, uses config.OPENROUTER_MODEL.
    response_language : str | None
        Force response in this language (e.g. "Japanese"). If None, mirrors question language.

    Returns
    -------
    Generator[str, None, None] | str
        Streamed tokens or full answer string.
    """
    # Re-read config.py fresh every call — bypasses Python's module cache so
    # any model change in config.py takes effect immediately without restart.
    importlib.reload(config)

    _model = model or config.OPENROUTER_MODEL  # sidebar override or config default

    # Build system prompt — inject language override if specified
    system_content = SYSTEM_PROMPT
    if response_language:
        system_content += (
            f"\n\nIMPORTANT: You MUST respond entirely in {response_language}, "
            f"regardless of the language of the question or the context."
        )

    client = _get_client()
    user_prompt = _build_user_prompt(question, context_chunks)

    messages = [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": user_prompt},
    ]

    logger.info(
        f"Calling OpenRouter model='{_model}' "
        f"with {len(context_chunks)} context chunks, stream={stream}"
    )

    response = client.chat.completions.create(
        model=_model,
        messages=messages,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=0.4,
        stream=stream,
        extra_headers={
            "HTTP-Referer": "https://github.com/rag-poc",
            "X-Title": "RAG Marketing Chatbot POC",
        },
    )

    if stream:
        def _token_generator() -> Generator[str, None, None]:
            got_any = False
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                # Prefer regular content; fall back to reasoning_content for
                # thinking models (e.g. Nemotron) that put output there.
                text = delta.content
                if not text:
                    text = getattr(delta, "reasoning_content", None)
                if text:
                    got_any = True
                    yield text
            if not got_any:
                # Model returned empty stream — surface a helpful message
                yield (
                    "I received an empty response from the model. "
                    "Please try selecting a different model from the sidebar "
                    "(e.g. Poolside Laguna or Llama)."
                )
        return _token_generator()
    else:
        return response.choices[0].message.content or ""


def format_sources(context_chunks: list[dict]) -> str:
    """Format source attribution as a markdown string."""
    if not context_chunks:
        return ""
    seen = set()
    lines = ["**📎 Sources used:**"]
    for chunk in context_chunks:
        key = (chunk["source"], chunk.get("ts_label", ""))
        if key not in seen:
            seen.add(key)
            ts = f" _(@ {chunk['ts_label']})_" if chunk.get("ts_label") else ""
            lines.append(f"- 🎬 `{chunk['source']}`{ts}")
    return "\n".join(lines)
