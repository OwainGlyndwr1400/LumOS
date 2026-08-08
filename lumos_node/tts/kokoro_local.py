"""Standalone Kokoro TTS via kokoro-onnx — bypasses LM Studio's TTS engine.

First call downloads the ONNX model (~310MB) and voices file (~6MB) to
~/.cache/lumos_kokoro/ and caches them. Subsequent calls reuse the loaded
model in-process (~few seconds CPU latency per response).
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path
from typing import Any

from ..log import get_logger

log = get_logger(__name__)

# Kokoro v1.0 has a hard 510-PHONEME input limit per call — phonemes, not
# characters. Plain prose runs ~1.0-1.2 phonemes/char, but emoji/markdown/table
# text (a GeoSentinel ping) expands far past that, so 400 chars could exceed 510
# phonemes and the chunk was DROPPED from the audio. Chunk conservatively and
# recursively halve anything Kokoro still rejects (_create_with_resplit), so no
# content is lost regardless of density.
KOKORO_CHUNK_CHARS = 280

# kokoro-onnx v1.0 release files (thewh1teagle/kokoro-onnx on GitHub).
MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/kokoro-v1.0.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/voices-v1.0.bin"
)

_kokoro_instance: Any = None


def _cache_dir() -> Path:
    p = Path.home() / ".cache" / "lumos_kokoro"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_models() -> tuple[Path, Path]:
    cache = _cache_dir()
    model_path = cache / "kokoro-v1.0.onnx"
    voices_path = cache / "voices-v1.0.bin"

    if not model_path.exists():
        log.info("kokoro.downloading_model", url=MODEL_URL, target=str(model_path))
        urllib.request.urlretrieve(MODEL_URL, model_path)
        log.info("kokoro.model_downloaded", bytes=model_path.stat().st_size)

    if not voices_path.exists():
        log.info("kokoro.downloading_voices", url=VOICES_URL, target=str(voices_path))
        urllib.request.urlretrieve(VOICES_URL, voices_path)
        log.info("kokoro.voices_downloaded", bytes=voices_path.stat().st_size)

    return model_path, voices_path


def is_available() -> bool:
    """True iff kokoro-onnx + soundfile are importable."""
    try:
        import kokoro_onnx  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        return False
    return True


def _get_kokoro() -> Any:
    global _kokoro_instance
    if _kokoro_instance is None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:
            raise RuntimeError(
                "kokoro-onnx not installed. Run: "
                "uv pip install kokoro-onnx soundfile"
            ) from e
        model_path, voices_path = _ensure_models()
        log.info("kokoro.initializing", model=str(model_path))
        _kokoro_instance = Kokoro(str(model_path), str(voices_path))
    return _kokoro_instance


def _split_for_kokoro(text: str, max_chars: int = KOKORO_CHUNK_CHARS) -> list[str]:
    """Sentence-aware chunking respecting Kokoro's per-call phoneme limit.

    Strategy:
      1. Split by sentence-ending punctuation (.!?).
      2. If a single sentence still exceeds max_chars, split by clause
         punctuation (,;:).
      3. If a clause still exceeds, force-split by character count.
      4. Greedily accumulate chunks up to max_chars.
    """
    text = text.strip()
    if not text:
        return []

    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        # Sentence too long — split by clause punctuation.
        for clause in re.split(r"(?<=[,;:])\s+", sentence):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= max_chars:
                pieces.append(clause)
                continue
            # Clause still too long — force-split by chars at whitespace.
            words = clause.split()
            buf = ""
            for w in words:
                if len(buf) + 1 + len(w) > max_chars and buf:
                    pieces.append(buf)
                    buf = w
                else:
                    buf = (buf + " " + w) if buf else w
            if buf:
                pieces.append(buf)

    # Greedy merge of small pieces up to max_chars.
    chunks: list[str] = []
    current = ""
    for p in pieces:
        if not current:
            current = p
        elif len(current) + 1 + len(p) <= max_chars:
            current = current + " " + p
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def _midpoint_split(text: str) -> int:
    """Index near the middle of text, preferring a whitespace boundary so a word
    isn't cut in half."""
    mid = len(text) // 2
    for off in range(mid):
        if text[mid - off] == " ":
            return mid - off
        if mid + off < len(text) and text[mid + off] == " ":
            return mid + off
    return mid


def _create_with_resplit(
    kokoro: Any,
    text: str,
    voice: str,
    speed: float,
    lang: str,
    depth: int = 0,
) -> tuple[list, int | None, Exception | None]:
    """Synthesize one chunk, self-correcting for phoneme density.

    Kokoro's cap is 510 PHONEMES; character count only approximates that, so a
    chunk under the char cap can still overflow (emoji/table-dense text) and
    raise "index 510 is out of bounds". On that specific overflow we split the
    text in half and recurse, so the content is spoken instead of dropped.
    Returns (sample_arrays, sample_rate, last_error).
    """
    text = text.strip()
    if not text:
        return [], None, None
    try:
        samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        return [samples], sr, None
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        overflow = "out of bounds" in msg or "phoneme" in msg or "too long" in msg
        if overflow and depth < 6 and len(text) > 24:
            mid = _midpoint_split(text)
            left, right = text[:mid].strip(), text[mid:].strip()
            if left and right:
                log.info("kokoro.chunk_resplit", chars=len(text), depth=depth)
                lp, lsr, le = _create_with_resplit(kokoro, left, voice, speed, lang, depth + 1)
                rp, rsr, re_ = _create_with_resplit(kokoro, right, voice, speed, lang, depth + 1)
                return lp + rp, (lsr or rsr), (le or re_)
        log.warning(
            "kokoro.chunk_failed",
            chars=len(text),
            preview=text[:60],
            error=str(e),
        )
        return [], None, e


def synthesize(
    text: str,
    *,
    voice: str = "af_bella",
    speed: float = 1.0,
    lang: str = "en-us",
) -> tuple[bytes, str]:
    """Return (wav_bytes, mime_type) for the given text.

    Splits text into Kokoro-safe chunks, synthesizes each, concatenates audio.
    """
    if not text.strip():
        return b"", "audio/wav"
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            "soundfile/numpy not installed. Run: uv pip install soundfile numpy"
        ) from e

    kokoro = _get_kokoro()
    chunks = _split_for_kokoro(text)
    if not chunks:
        return b"", "audio/wav"

    log.info("kokoro.synthesize", chunks=len(chunks), total_chars=len(text), voice=voice)

    samples_list: list = []
    sample_rate = 24000
    last_error: Exception | None = None
    for chunk in chunks:
        parts, sr, err = _create_with_resplit(kokoro, chunk, voice, speed, lang)
        if err is not None:
            last_error = err
        if parts:
            samples_list.extend(parts)
            if sr:
                sample_rate = sr

    if not samples_list:
        # Raise instead of returning empty audio: a silent 0-byte WAV renders in
        # the HUD as an unplayable blob / "Failed to fetch" with no clue why. The
        # usual cause is an invalid or empty voice name (kokoro asserts the voice
        # exists). Surface it so /speak returns a real 502 with the reason.
        raise RuntimeError(
            f"kokoro produced no audio — all {len(chunks)} chunk(s) failed"
            + (f" (voice={voice!r}; last error: {last_error})" if last_error else "")
        )

    combined = np.concatenate(samples_list) if len(samples_list) > 1 else samples_list[0]
    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV")
    return buf.getvalue(), "audio/wav"


def prewarm() -> dict[str, Any]:
    """Force model download + load. Returns a small status dict."""
    if not is_available():
        return {"available": False, "reason": "kokoro-onnx not installed"}
    model_path, voices_path = _ensure_models()
    _ = _get_kokoro()
    return {
        "available": True,
        "model": str(model_path),
        "voices": str(voices_path),
        "model_bytes": model_path.stat().st_size,
        "voices_bytes": voices_path.stat().st_size,
    }
