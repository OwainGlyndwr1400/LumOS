"""NVIDIA Magpie TTS via the hosted NVCF Riva endpoint (Overdrive-grade voice).

Unlike Kokoro (local ONNX) and LM Studio (OpenAI-compatible HTTP), the hosted
Magpie models are Riva NIMs addressed over **gRPC** at grpc.nvcf.nvidia.com:443
by a `function-id` — there is no OpenAI `/v1/audio/speech` shape and no pure-HTTP
path to the cloud model. So this provider needs the `nvidia-riva-client` package
(imported as `riva.client`). It stays optional: `is_available()` guards every
entry point, and the /speak route returns a clear 503 when it's missing.

The service returns raw 16-bit mono LPCM in `resp.audio`; we wrap it into a WAV
container with the stdlib `wave` module (no soundfile dependency needed here).

Requires LUMOS_NVIDIA_API_KEY (the same free build.nvidia.com key Overdrive uses).
"""

from __future__ import annotations

import concurrent.futures
import io
import re
import wave
from typing import Any

from ..log import get_logger

log = get_logger(__name__)

# Magpie's HARD cap is ~400 tokens of ITS OWN sequence units — NOT characters.
# Dense text (a table row of dates/coords/symbols) expands ~2.7x, so a 297-char
# chunk measured 708 tokens and was rejected; prose is ~1.3x. No single char cap
# is safe for both, so we chunk conservatively (below ~2.7x*140 < 400 for dense
# content) AND recursively halve any chunk the server still rejects as too long
# (_synthesize_chunk), so nothing is dropped regardless of density. A separate
# wall-clock timeout guards against the OTHER failure mode: a too-LONG input used
# to HANG the NIM indefinitely (64 chars OK, 791 froze).
_MAGPIE_CHUNK_CHARS = 140

# Per-chunk wall-clock deadline. A chunk that hasn't returned by this is treated
# as a failure (logged + fail-fast) rather than hanging the synthesis request.
_CHUNK_TIMEOUT_S = 30.0

# 16-bit mono PCM — matches AudioEncoding.LINEAR_PCM from the hosted Magpie NIM.
_SAMPLE_WIDTH_BYTES = 2
_CHANNELS = 1

# Cache one SpeechSynthesisService per (uri, function_id, api_key) so we reuse
# the gRPC channel across calls instead of reopening it every /speak.
_service_cache: dict[tuple[str, str, str], Any] = {}


def is_available() -> bool:
    """True iff the nvidia-riva-client package is importable."""
    try:
        import riva.client  # noqa: F401
    except ImportError:
        return False
    return True


def _get_service(uri: str, function_id: str, api_key: str) -> Any:
    key = (uri, function_id, api_key)
    svc = _service_cache.get(key)
    if svc is not None:
        return svc
    try:
        import riva.client
    except ImportError as e:  # pragma: no cover - guarded by is_available()
        raise RuntimeError(
            "nvidia-riva-client not installed. Run: "
            "uv pip install nvidia-riva-client"
        ) from e

    auth = riva.client.Auth(
        uri=uri,
        use_ssl=True,
        metadata_args=[
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ],
    )
    svc = riva.client.SpeechSynthesisService(auth)
    _service_cache[key] = svc
    log.info("magpie.service_init", uri=uri, function_id=function_id[:8] + "…")
    return svc


def _linear_pcm_encoding() -> Any:
    """Resolve AudioEncoding.LINEAR_PCM across nvidia-riva-client versions."""
    try:
        from riva.client import AudioEncoding

        return AudioEncoding.LINEAR_PCM
    except Exception:  # noqa: BLE001 - fall back to the proto module path
        from riva.client.proto.riva_audio_pb2 import AudioEncoding

        return AudioEncoding.LINEAR_PCM


def _split_for_magpie(text: str, max_chars: int = _MAGPIE_CHUNK_CHARS) -> list[str]:
    """Sentence-aware chunking: split on sentence enders, then greedily merge
    up to max_chars. Over-long single sentences are hard-split on whitespace."""
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
        words, buf = sentence.split(), ""
        for w in words:
            if len(buf) + 1 + len(w) > max_chars and buf:
                pieces.append(buf)
                buf = w
            else:
                buf = (buf + " " + w) if buf else w
        if buf:
            pieces.append(buf)

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


def _synthesize_with_timeout(
    service: Any,
    text: str,
    voice: str,
    language: str,
    sample_rate_hz: int,
    encoding: Any,
) -> Any:
    """Run Riva's blocking synthesize with a wall-clock deadline.

    Riva's synthesize can HANG (not raise) on an over-limit input. A throwaway
    single-worker executor lets us bound it: on timeout we raise TimeoutError
    (caught per-chunk) and abandon the stuck worker via shutdown(wait=False)
    instead of blocking — so the request degrades to a logged failure rather
    than freezing indefinitely.
    """
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(
            service.synthesize,
            text,
            voice,
            language,
            sample_rate_hz=sample_rate_hz,
            encoding=encoding,
        )
        return fut.result(timeout=_CHUNK_TIMEOUT_S)
    finally:
        ex.shutdown(wait=False)


def _midpoint_split(text: str) -> int:
    """Index near the middle of text to split at, preferring a whitespace
    boundary so a word isn't cut in half."""
    mid = len(text) // 2
    for off in range(mid):
        if text[mid - off] == " ":
            return mid - off
        if mid + off < len(text) and text[mid + off] == " ":
            return mid + off
    return mid  # no space found — hard split


def _synthesize_chunk(
    service: Any,
    text: str,
    voice: str,
    language: str,
    sample_rate_hz: int,
    encoding: Any,
    *,
    depth: int = 0,
) -> bytes:
    """Synthesize one chunk to raw PCM, self-correcting for token density.

    Magpie's real cap is ~400 tokens of ITS OWN units (not characters); dense
    number/symbol text can overflow it even under the char cap. On that specific
    "too long" rejection we split the text in half and recurse — so no content is
    ever dropped, regardless of density. Returns b"" only if a piece is genuinely
    unrecoverable. Re-raises TimeoutError so synthesize() can fail-fast on the
    OTHER failure mode (a hang)."""
    text = text.strip()
    if not text:
        return b""
    try:
        resp = _synthesize_with_timeout(service, text, voice, language, sample_rate_hz, encoding)
        return getattr(resp, "audio", b"") or b""
    except concurrent.futures.TimeoutError:
        raise  # hang — surface to synthesize() so it aborts the remaining chunks
    except Exception as e:  # noqa: BLE001
        over_limit = "maximum sequence length" in str(e) or "longer than" in str(e)
        if over_limit and depth < 6 and len(text) > 24:
            mid = _midpoint_split(text)
            left, right = text[:mid].strip(), text[mid:].strip()
            if left and right:
                log.info("magpie.chunk_resplit", chars=len(text), depth=depth)
                return _synthesize_chunk(
                    service, left, voice, language, sample_rate_hz, encoding, depth=depth + 1
                ) + _synthesize_chunk(
                    service, right, voice, language, sample_rate_hz, encoding, depth=depth + 1
                )
        log.warning(
            "magpie.chunk_failed",
            chars=len(text),
            preview=text[:60],
            error=f"{type(e).__name__}: {e}",
        )
        return b""


def _pcm_to_wav(pcm: bytes, sample_rate_hz: int) -> bytes:
    """Wrap raw 16-bit mono LPCM bytes in a WAV container (header patched on
    close so players read the correct length)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm)
    return buf.getvalue()


def synthesize(
    text: str,
    *,
    voice: str,
    api_key: str,
    function_id: str,
    uri: str,
    language: str = "en-US",
    sample_rate_hz: int = 22050,
) -> tuple[bytes, str]:
    """Return (wav_bytes, "audio/wav") for text via hosted Magpie TTS.

    Blocking gRPC — call from a thread (the /speak route uses asyncio.to_thread).
    Raises on total failure so /speak surfaces a real 502 instead of silent
    empty audio (mirrors the kokoro_local contract).
    """
    if not text.strip():
        return b"", "audio/wav"

    service = _get_service(uri, function_id, api_key)
    encoding = _linear_pcm_encoding()
    chunks = _split_for_magpie(text)
    if not chunks:
        return b"", "audio/wav"

    log.info(
        "magpie.synthesize",
        chunks=len(chunks),
        total_chars=len(text),
        voice=voice,
        sample_rate=sample_rate_hz,
    )

    pcm_parts: list[bytes] = []
    last_error: Exception | None = None
    for chunk in chunks:
        try:
            audio = _synthesize_chunk(service, chunk, voice, language, sample_rate_hz, encoding)
        except concurrent.futures.TimeoutError as e:
            # A timeout means this input hit the server's hang threshold — the
            # remaining chunks would stall identically, so fail fast.
            last_error = e
            log.warning("magpie.aborted_on_timeout", error=str(e))
            break
        if audio:
            pcm_parts.append(audio)

    if not pcm_parts:
        raise RuntimeError(
            f"Magpie produced no audio — all {len(chunks)} chunk(s) failed"
            + (f" (voice={voice!r}; last error: {last_error})" if last_error else "")
        )

    return _pcm_to_wav(b"".join(pcm_parts), sample_rate_hz), "audio/wav"
