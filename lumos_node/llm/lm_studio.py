from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel

from ..config import get_settings, overdrive_on
from ..log import get_logger

log = get_logger(__name__)

# ── Per-(endpoint, model) API quirks, LEARNED from the provider's own 400 ─────
# Newer OpenAI reasoning models (gpt-5.x, o-series) reject the classic sampling
# payload: `max_tokens` must be `max_completion_tokens`, and temperature/top_p
# must be left at their defaults. Rather than hardcode model-name patterns that
# go stale every release, we read what the API tells us, apply it, and remember
# it for the process — so the retry happens at most once per model.
_QUIRK_COMPLETION_TOKENS = "max_completion_tokens"
_QUIRK_NO_SAMPLING = "no_sampling"
_QUIRK_NO_REASONING = "reasoning_effort_none"
# /v1/responses only: a NON-reasoning model (gpt-4o) rejects the `reasoning`
# block outright — drop it and the same responses call works fine.
_QUIRK_NO_REASONING_BLOCK = "no_reasoning_block"
_model_quirks: dict[tuple[str, str], set[str]] = {}

# Quirks that cost real capability get an explicit note — a silent downgrade is
# worse than a loud one.
_QUIRK_NOTES: dict[str, str] = {
    _QUIRK_NO_REASONING: (
        "model refuses function tools on /v1/chat/completions unless reasoning is "
        "off. Tools are core to Lumos, so REASONING IS DISABLED for this model. "
        "To keep reasoning you need the /v1/responses API, or use a tool-native "
        "model (e.g. gpt-4o) instead."
    ),
}


def _learn_quirk(text: str) -> str | None:
    """Map a provider 400 body to a known payload quirk, or None if unrelated."""
    low = (text or "").lower()
    rejected = (
        "does not support" in low
        or "unsupported value" in low
        or "unsupported parameter" in low
        or "not supported" in low
    )
    # OpenAI names the replacement ("Use 'max_completion_tokens' instead"), but
    # accept a bare "'max_tokens' is not supported" too.
    if "max_completion_tokens" in low or (rejected and "max_tokens" in low):
        return _QUIRK_COMPLETION_TOKENS
    # gpt-5.6-terra & friends refuse function tools unless reasoning is off:
    # "Function tools with reasoning_effort are not supported ... set
    #  reasoning_effort to 'none'". Lumos always sends tools, so without this
    # every single turn 400s on those models.
    # "reasoning.effort" (dot) is the /responses payload field — a non-reasoning
    # model rejects the whole block. "reasoning_effort" (underscore) is the
    # chat/completions param, which means the OPPOSITE fix (pin it to 'none').
    if "reasoning.effort" in low or ("reasoning" in low and "unsupported parameter" in low):
        return _QUIRK_NO_REASONING_BLOCK
    if "reasoning_effort" in low:
        return _QUIRK_NO_REASONING
    if rejected and ("temperature" in low or "top_p" in low):
        return _QUIRK_NO_SAMPLING
    return None


def _apply_quirks(payload: dict[str, Any], quirks: set[str]) -> dict[str, Any]:
    """Reshape a chat payload for a model with known parameter restrictions."""
    if _QUIRK_COMPLETION_TOKENS in quirks and "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
    if _QUIRK_NO_SAMPLING in quirks:
        payload.pop("temperature", None)
        payload.pop("top_p", None)
    if _QUIRK_NO_REASONING in quirks:
        payload["reasoning_effort"] = "none"
    return payload


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


# ── OpenAI /v1/responses translation (reasoning models WITH tools) ────────────
# gpt-5.6-terra & the o-series refuse function tools on /chat/completions unless
# reasoning is off. /v1/responses allows tools AND reasoning together but uses a
# different request/response shape. We translate at the CLIENT boundary so chat()
# still returns the same {role, content, tool_calls, usage} dict the tool loop
# already consumes — nothing downstream changes.


def _to_responses_input(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Chat messages -> /responses `input` items. Assistant tool_calls become
    function_call items; role='tool' results become function_call_output items."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({
                "type": "function_call_output",
                "call_id": m.tool_call_id or "",
                "output": m.content if isinstance(m.content, str) else str(m.content),
            })
        elif m.role == "assistant" and m.tool_calls:
            if m.content:
                out.append({"role": "assistant", "content": m.content})
            for tc in m.tool_calls:
                fn = tc.get("function") or {}
                out.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                })
        elif m.content is not None:
            out.append({"role": m.role, "content": m.content})
    return out


def _to_responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Chat tool schema {type:function, function:{...}} -> FLAT responses schema."""
    flat: list[dict[str, Any]] = []
    for t in tools or []:
        fn = t.get("function") or {}
        flat.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return flat


def _from_responses_output(data: dict[str, Any]) -> dict[str, Any]:
    """/responses output -> the chat-completions-shaped message dict the tool loop
    consumes: concatenate output_text, map function_call -> tool_calls, map usage."""
    items = data.get("output") or []
    if not items and not data.get("output_text"):
        raise RuntimeError(f"responses returned no output: {data.get('error') or data}")
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for it in items:
        itype = it.get("type")
        if itype == "message":
            for c in it.get("content") or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    text_parts.append(c["text"])
        elif itype == "function_call":
            tool_calls.append({
                "id": it.get("call_id") or it.get("id") or "",
                "type": "function",
                "function": {"name": it.get("name", ""), "arguments": it.get("arguments", "{}")},
            })
        # "reasoning" items carry the CoT summary — deliberately not surfaced.
    content = "".join(text_parts) or (data.get("output_text") or None)
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    u = data.get("usage") or {}
    if u:
        msg["usage"] = {
            "prompt_tokens": u.get("input_tokens"),
            "completion_tokens": u.get("output_tokens"),
            "total_tokens": u.get("total_tokens"),
        }
    msg["finish_reason"] = data.get("status") or "stop"
    return msg


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
        # Route through /v1/responses only when explicitly enabled AND actually
        # pointed at OpenAI — so a stray flag can never redirect a local call.
        self._use_responses = (
            settings.openai_use_responses_api and "openai.com" in self.base_url
        )
        self._reasoning_effort = settings.openai_reasoning_effort

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

    async def _chat_responses(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Non-streaming call via OpenAI /v1/responses — reasoning stays ON while
        tools work. Returns the SAME message dict shape as chat() (see the
        _from_responses_output translator), so the tool loop is unchanged."""
        body: dict[str, Any] = {
            "model": model,
            "input": _to_responses_input(messages),
            "reasoning": {"effort": self._reasoning_effort},
            "max_output_tokens": max_tokens or get_settings().openai_max_tokens,
        }
        rtools = _to_responses_tools(tools)
        if rtools:
            body["tools"] = rtools
            body["tool_choice"] = "auto"
        # Same learn-from-the-400 loop as the chat/completions path: a NON-reasoning
        # model (gpt-4o) served via /responses rejects the `reasoning` block, so
        # drop it and retry. Cached per (endpoint, model) — one retry, once.
        quirks = _model_quirks.setdefault((self.base_url, model), set())
        for _ in range(3):
            if _QUIRK_NO_REASONING_BLOCK in quirks:
                body.pop("reasoning", None)
            resp = await self._client.post(
                f"{self.base_url}/responses",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            if resp.status_code != 400:
                break
            learned = _learn_quirk(resp.text)
            if learned is None or learned in quirks:
                break
            quirks.add(learned)
            log.info(
                "llm.quirk_learned", model=model, quirk=learned, path="responses",
                note=_QUIRK_NOTES.get(learned),
            )
        resp.raise_for_status()
        return _from_responses_output(resp.json())

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
        # Reasoning-model path: tools + reasoning via /v1/responses. Structured
        # output (response_format) stays on chat/completions — it uses a different
        # shape there and isn't part of the tool/wake path.
        if self._use_responses and response_format is None:
            return await self._chat_responses(model, messages, tools, max_tokens)
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
        quirks = _model_quirks.setdefault((self.base_url, model), set())
        for _ in range(3):
            _apply_quirks(payload, quirks)
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            if resp.status_code != 400:
                break
            learned = _learn_quirk(resp.text)
            if learned is None or learned in quirks:
                break  # a 400 we can't fix — let raise_for_status surface it
            quirks.add(learned)
            log.info("llm.quirk_learned", model=model, quirk=learned)
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
        # Responses path, first cut: one non-streaming /responses call, emitted as
        # a single delta + finished chunk. With tools always on, the operator path
        # almost always takes the fast-path reveal anyway; correctness over
        # typewriter here. (Real responses-SSE streaming is a later upgrade.)
        if self._use_responses:
            msg = await self._chat_responses(model, messages, None, max_tokens)
            text = msg.get("content") or ""
            if text:
                yield CompletionChunk(delta=text)
            yield CompletionChunk(finished=True, usage=msg.get("usage"))
            return
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
        quirks = _model_quirks.setdefault((self.base_url, model), set())
        for _ in range(3):
            _apply_quirks(payload, quirks)
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as resp:
                # A rejected payload 400s before any SSE body arrives — read the
                # provider's complaint, learn the quirk, and retry the stream.
                if resp.status_code == 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    learned = _learn_quirk(body)
                    if learned is not None and learned not in quirks:
                        quirks.add(learned)
                        log.info(
                    "llm.quirk_learned",
                    model=model,
                    quirk=learned,
                    note=_QUIRK_NOTES.get(learned),
                )
                        continue
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
            # Stream consumed (with or without a [DONE]) — never retry a
            # successful request; the retry loop exists only for payload 400s.
            return


def _mime_for(fmt: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/ogg",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "pcm": "audio/pcm",
    }.get(fmt.lower(), "audio/mpeg")
