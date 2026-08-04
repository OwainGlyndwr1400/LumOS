from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel

from ..config import get_settings, overdrive_on


class ChatMessage(BaseModel):
    role: str
    # Content may be a plain string OR a list of OpenAI multimodal parts
    # (e.g. [{"type":"text","text":...},{"type":"image_url","image_url":{...}}]).
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class CompletionChunk(BaseModel):
    delta: str = ""
    finished: bool = False
    usage: dict[str, Any] | None = None


class LMStudioClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.lm_studio_base_url).rstrip("/")
        self.api_key = api_key or settings.lm_studio_api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        # Short per-call timeout: /models is a management poll, so it must not
        # inherit the 600s chat default (a stalled LM Studio would hang callers
        # — e.g. the swap-orchestration pre-check — for up to 10 minutes).
        resp = await self._client.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        # Embeddings ALWAYS hit the LOCAL endpoint — NVIDIA Overdrive only swaps
        # the CHAT brain. NVIDIA's cloud API has no matching /embeddings model and
        # the FAISS index is local-BGE dims, so routing embeds to the cloud 404s
        # and would corrupt retrieval. Resolve per-call so this holds regardless
        # of how this client's base_url was set or the current Overdrive state.
        from ..config import local_embeddings_endpoint

        embed_base, embed_key = local_embeddings_endpoint()
        resp = await self._client.post(
            f"{embed_base.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {embed_key}"},
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        data.sort(key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in data]

    async def speak(
        self,
        text: str,
        *,
        model: str,
        voice: str,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> tuple[bytes, str]:
        """Synthesize speech via /v1/audio/speech. Returns (audio_bytes, mime_type)."""
        if not text.strip():
            return b"", _mime_for(response_format)
        payload: dict[str, Any] = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
            "speed": speed,
        }
        resp = await self._client.post(
            f"{self.base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type", _mime_for(response_format))

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Single non-streaming chat completion. Returns the raw message dict
        from LM Studio: {role, content, tool_calls?}.

        `response_format` accepts OpenAI structured-output schemas, e.g.
            {"type": "json_schema", "json_schema": {"name": "X", "schema": {...}}}
        LM Studio enforces the schema and guarantees valid JSON in `content`.

        `chat_template_kwargs` (Phase 33) — extra params forwarded to the model's
        Jinja chat template. Standard de-facto key across Qwen3.5 / Gemma 4 thinking
        models is `enable_thinking: bool`. Models that don't recognize a key ignore
        it harmlessly, so this is safe to pass even on non-thinking models.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format is not None:
            payload["response_format"] = response_format
        # chat_template_kwargs is an LM-Studio-only Jinja field (thinking-mode
        # control). Cloud OpenAI-compat endpoints reject unknown fields — Gemini
        # 400s "Unknown name chat_template_kwargs". Only send it to local.
        if chat_template_kwargs and not overdrive_on():
            payload["chat_template_kwargs"] = chat_template_kwargs
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            # A 200 with no usable choices is an error envelope (e.g. an Overdrive
            # cloud provider returning {"error": ...} at HTTP 200, or LM Studio
            # surfacing a model error) — fail with a clear message instead of a
            # cryptic KeyError('choices') that crashes the whole turn.
            raise RuntimeError(
                f"chat completion returned no choices: {data.get('error') or data}"
            )
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            raise RuntimeError("chat completion first choice had no message object")
        # Surface top-level usage on the message dict so a non-streaming caller can
        # capture token counts (e.g. chat.py's reuse-without-restream fast path).
        if data.get("usage") is not None:
            msg.setdefault("usage", data["usage"])
        # Surface finish_reason so callers can detect truncated/degenerate
        # generations ("length") without re-parsing the raw response.
        msg.setdefault("finish_reason", choices[0].get("finish_reason"))
        return msg

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        top_p: float | None = None,
    ) -> AsyncIterator[CompletionChunk]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # chat_template_kwargs is an LM-Studio-only Jinja field (thinking-mode
        # control). Cloud OpenAI-compat endpoints reject unknown fields — Gemini
        # 400s "Unknown name chat_template_kwargs". Only send it to local.
        if chat_template_kwargs and not overdrive_on():
            payload["chat_template_kwargs"] = chat_template_kwargs
        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        ) as resp:
            resp.raise_for_status()
            import json
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    yield CompletionChunk(finished=True)
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue  # partial/keepalive line — skip, don't kill the stream
                usage = obj.get("usage")
                if usage:
                    yield CompletionChunk(usage=usage)
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {}).get("content", "") or ""
                if delta:
                    yield CompletionChunk(delta=delta)
                # Note: do NOT yield finished=True on finish_reason; the usage
                # chunk arrives AFTER the finish_reason chunk per the OpenAI
                # spec. We rely on `[DONE]` as the sole terminal marker.


def _mime_for(fmt: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/ogg",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "pcm": "audio/pcm",
    }.get(fmt.lower(), "audio/mpeg")
