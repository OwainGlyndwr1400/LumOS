"""Regression guard — dream-cycle compression must not let gpt-oss-20b run away.

Observed bug (2026-06-09): compress_chunk called the LLM with no token cap and in
thinking mode; gpt-oss-20b degenerated into a repetition loop ("Lost-2"/"core
truth"/"binding-energy" × 1000+ reasoning tokens), throwing Channel Errors and
disconnecting. Fix: hard max_tokens cap + enable_thinking=False + de-baited prompt.
These tests lock all three in.
"""

import asyncio
import json

from lumos_node import compression as comp


class _RecordingClient:
    """Stub LMStudioClient — records the chat() kwargs, returns a valid packet."""

    def __init__(self):
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "content": json.dumps(
                {
                    "summary_object": {"summary": "s", "key_points": ["a", "b"]},
                    "anchor_packet": {},
                    "compressed_operational_packet": "p",
                }
            )
        }


def test_compress_chunk_caps_tokens_and_disables_thinking():
    fake = _RecordingClient()
    asyncio.run(comp.compress_chunk("text to compress", model="m", client=fake))
    assert len(fake.calls) == 1
    call = fake.calls[0]
    # The two structural runaway-loop guards must be on the LLM call.
    assert call.get("max_tokens") == comp.COMPRESSION_MAX_TOKENS
    assert call.get("chat_template_kwargs") == {"enable_thinking": False}
    assert 1 <= comp.COMPRESSION_MAX_TOKENS <= 1024   # bounded + sane


class _BadJsonClient:
    """Stub that returns truncated/invalid JSON — the real flaky-model failure."""

    async def chat(self, **kwargs):
        return {"content": '{"summary_object": {"summary": "trunc'}  # unterminated


def test_compress_chunk_falls_back_to_extractive_on_bad_json():
    out = asyncio.run(
        comp.compress_chunk(
            "Erydir studies the Recursive Harmonic Codex with constants like phi.",
            model="m",
            client=_BadJsonClient(),
        )
    )
    # No longer None on bad JSON — the chunk still compresses via the heuristic.
    assert out is not None
    assert "summary_object" in out and "anchor_packet" in out
    assert "compressed_operational_packet" in out and "tokens" in out
    assert out["anchor_packet"].get("source_type") == "extractive"


def test_compression_prompt_debaited():
    # The exact phrases gpt-oss-20b looped on must be gone from the prompt.
    assert "Lost-2" not in comp.COMPRESSION_PROMPT
    assert "binding-energy" not in comp.COMPRESSION_PROMPT


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
