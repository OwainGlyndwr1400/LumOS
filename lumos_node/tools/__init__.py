"""Tool registry + OpenAI tool-schema generation for Lumos.

Tools are exposed to LM Studio via the `tools` parameter (NOT injected into
the system prompt — runtime isolation stays intact). The model decides when
to invoke them; we execute, append `tool` role messages, loop until the
model returns content with no more tool_calls.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..log import get_logger

log = get_logger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_registry: dict[str, Tool] = {}


def register(
    name: str, description: str, parameters: dict[str, Any]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _registry[name] = Tool(
            name=name, description=description, parameters=parameters, func=func
        )
        return func

    return decorator


def get_registry() -> dict[str, Tool]:
    return _registry


def get_schemas() -> list[dict[str, Any]]:
    return [t.to_openai_schema() for t in _registry.values()]


def get_schemas_filtered(names: list[str] | set[str]) -> list[dict[str, Any]]:
    """Phase 35 — return OpenAI schemas only for the given tool names.

    Used by `tool_router.select_tools()` to send a turn-specific subset to
    LM Studio instead of the full ~36-tool catalog. Preserves registry order
    for prefix-cache stability (same subset → same byte-stable prefix across
    repeat queries). Unknown names are silently skipped.
    """
    wanted = set(names)
    return [
        t.to_openai_schema() for n, t in _registry.items() if n in wanted
    ]


def get_tool_names() -> list[str]:
    """Sorted list of all registered tool names. Used by the router for
    category-membership validation and by HUD/CLI for introspection."""
    return sorted(_registry.keys())


def _type_ok(value: Any, expected: str | list[str]) -> bool:
    allowed = expected if isinstance(expected, list) else [expected]
    for kind in allowed:
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "number" and isinstance(value, int | float) and not isinstance(value, bool):
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
        if kind == "null" and value is None:
            return True
    return False


def _validate_args(tool: Tool, args: dict[str, Any]) -> str | None:
    """Small JSON-schema subset validator for our OpenAI/MCP tool schemas."""
    schema = tool.parameters or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    missing = sorted(k for k in required if k not in args)
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"

    unknown = sorted(k for k in args if k not in props)
    if unknown:
        return f"unknown argument(s): {', '.join(unknown)}"

    for key, value in args.items():
        spec = props.get(key) or {}
        expected_type = spec.get("type")
        if expected_type and not _type_ok(value, expected_type):
            return f"argument '{key}' has wrong type"
        if isinstance(value, str):
            max_len = spec.get("maxLength")
            min_len = spec.get("minLength")
            if isinstance(max_len, int) and len(value) > max_len:
                return f"argument '{key}' is too long"
            if isinstance(min_len, int) and len(value) < min_len:
                return f"argument '{key}' is too short"
        if isinstance(value, int | float) and not isinstance(value, bool):
            minimum = spec.get("minimum")
            maximum = spec.get("maximum")
            if isinstance(minimum, int | float) and value < minimum:
                return f"argument '{key}' is below minimum"
            if isinstance(maximum, int | float) and value > maximum:
                return f"argument '{key}' is above maximum"
    return None


async def execute_tool(
    name: str, args: dict[str, Any], allowed_tools: set[str] | None = None
) -> str:
    """Run a registered tool. Returns a JSON-serializable string suitable
    for inclusion as a tool-role message content.

    `allowed_tools`, when provided, is a HARD execution allowlist (defense in
    depth for autonomous/passive turns — "autonomy ends at speaking"). A tool
    whose name is not in the set is refused HERE, at the execution boundary,
    regardless of which schemas the model was shown — so the safety guarantee
    never relies on model compliance. Mirrors the MCP server's allowlist check.
    """
    if allowed_tools is not None and name not in allowed_tools:
        log.warning("tools.blocked_not_allowed", tool=name)
        return json.dumps({
            "error": (
                f"tool '{name}' blocked: it was not offered on this turn "
                "(execution allowlist — routing subset or passive/observe-only boundary)"
            ),
            "tool": name,
        })
    tool = _registry.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    validation_error = _validate_args(tool, args)
    if validation_error is not None:
        return json.dumps({"error": validation_error, "tool": name})
    try:
        result = tool.func(**args)
        if inspect.isawaitable(result):
            result = await result
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e), "tool": name})
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"unserializable result: {e}"})


# Import side-effects: each module registers its tools at import time.
from . import (  # noqa: E402, F401
    file_tools,  # noqa: E402, F401
    forecast_tools,  # noqa: E402, F401  # anticipatory look-ahead (passive)
    forge_tools,  # noqa: E402, F401  # forge_verify — Forge-only, gated exec runner
    git_tools,  # noqa: E402, F401
    intel_tools,  # noqa: E402, F401  # Aether Scope global-intel layer
    memory_tools,  # noqa: E402, F401
    python_tools,  # noqa: E402, F401
    skill_tools,  # noqa: E402, F401
    task_tools,  # noqa: E402, F401
    telemetry_tools,  # noqa: E402, F401
    temporal_tools,  # noqa: E402, F401
    time_tools,  # noqa: E402, F401
    vault_tools,  # noqa: E402, F401  # Obsidian vault-graph — associative 3rd memory
    watch_tools,  # noqa: E402, F401  # custom watches (operator-only)
    web_tools,  # noqa: E402, F401
)
