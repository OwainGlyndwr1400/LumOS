"""Forge Mode (Phase 45) — operator-initiated coding sessions.

`lumos forge "fix the HUD stale-badge bug"` runs Lumos as a bounded coding
agent: it plans, edits files in a git workspace, runs the tests/linter via the
allow-listed forge_verify runner, READS the failures, and iterates until green
— then checkpoint-commits. This is the run→observe→fix loop that makes it a
coding agent rather than a code suggester (the Claude Code / Codex shape),
running entirely on the local brain.

Deliberately STANDALONE — it reuses LMStudioClient + the tool registry
directly and never touches chat.py, so the operator's three handwritten
persona prompts are untouched and Forge gets its own system prompt, its own
(much higher) iteration budget, and its own locked-down tool set.

Rails (all enforced by machinery Forge doesn't own, so it can't bypass them):
  • Writes: file tools honor LUMOS_TOOL_ALLOWED_PATHS; git tools honor
    LUMOS_GIT_WORKSPACES. Forge operates on one validated workspace.
  • Exec: only forge_verify, only allow-listed commands (see forge_tools).
  • Push: git_push / gh_create_pr are NOT in FORGE_TOOLS — a session
    physically cannot push. (forge_allow_push is reserved for a future opt-in.)
  • Autonomy: Forge is operator-initiated only. Wakes never enter it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .llm.lm_studio import ChatMessage, LMStudioClient
from .log import get_logger
from .tools import execute_tool, get_schemas_filtered

log = get_logger(__name__)

# Forge's tool set — file edit + git READ/stage/commit + scratch python + the
# verify runner + memory recall. NO git_push, NO gh_create_pr (never-push rule),
# NO web/telemetry (a coding session shouldn't reach the outside world).
FORGE_TOOLS: frozenset[str] = frozenset({
    "read_file", "write_file", "append_file", "list_files", "list_allowed_paths",
    "git_status", "git_diff", "git_log", "git_branch", "git_add", "git_commit",
    "run_python", "forge_verify",
    "search_memory", "search_knowledge",
})

# Re-inject the plan file this often so a small local model doesn't lose the
# thread over a long session (context-window discipline — the same lesson as
# the LM Studio template incident: never assume the model retains everything).
_PLAN_REINJECT_EVERY = 8
# Keep the message list bounded: system + original task + the most recent N.
_KEEP_RECENT = 30

EventFn = Callable[[dict[str, Any]], None] | Callable[[dict[str, Any]], Awaitable[None]]

_FORGE_SYSTEM = """You are Lumos in FORGE MODE — a focused, local coding agent working on the operator's behalf.

You are working inside ONE git workspace:
  {workspace}

THE TASK:
{task}

HOW YOU WORK (a disciplined loop, like a careful engineer):
1. PLAN FIRST. Write a short numbered plan to `{plan_file}` using write_file. Keep it updated as you learn — it is your memory across steps.
2. INVESTIGATE before editing: read the relevant files (read_file, list_files), understand the existing style, then make the smallest correct change.
3. EDIT with write_file / append_file. Match the surrounding code's conventions.
4. VERIFY every change: call forge_verify with a command like "pytest -q" or "ruff check .". READ the output. If it fails, read the failure, fix it, verify again. Do not claim success without a green verify.
5. CHECKPOINT: after a verify passes, git_add the changed files and git_commit with a clear message. Small green commits, not one big risky one.
6. When the task is DONE and verified, STOP calling tools and write a final summary: what you changed, which files, and the verify result.

HARD RULES:
- Only touch files inside the workspace above. Never try to push (you have no push tool — that is intentional; the operator pushes).
- If a verify command isn't allowed, use one that is (pytest / ruff / compile / npm build). Don't try to run arbitrary shell.
- If you get stuck after a few attempts on the same error, stop and explain what's blocking you rather than thrashing.
- Prefer correctness over cleverness. This is the operator's real system.

Begin by writing your plan."""


async def _emit(on_event: EventFn | None, event: dict[str, Any]) -> None:
    if on_event is None:
        log.info("forge.event", **{k: v for k, v in event.items() if k != "detail"})
        return
    res = on_event(event)
    if hasattr(res, "__await__"):
        await res  # type: ignore[func-returns-value]


def _resolve_workspace(explicit: str | None) -> Path:
    """Forge workspace: explicit arg → forge_workspace → first git_workspaces."""
    settings = get_settings()
    candidate = (explicit or settings.forge_workspace or "").strip()
    if not candidate:
        raw = settings.git_workspaces.strip()
        first = next((p.strip() for p in raw.split(",") if p.strip()), "")
        candidate = first
    if not candidate:
        raise ValueError(
            "No Forge workspace. Set LUMOS_FORGE_WORKSPACE or LUMOS_GIT_WORKSPACES "
            "to an absolute repo path and restart."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Forge workspace is not a directory: {path}")
    return path


def _swap_plan(settings: Any, model: str) -> tuple[bool, str | None]:
    """PURE — should we swap the brain for this session, and what to restore?

    Returns (should_swap, restore_target). Swap only when enabled AND the Forge
    model differs from the companion (model_light) — otherwise it's the same 9B
    and a swap would be a pointless eject/reload. restore_target is the
    companion brain wakes/chat default to, so it's warm again afterwards.
    """
    if not getattr(settings, "forge_swap_model", True):
        return False, None
    companion = settings.model_light
    if not model or model == companion:
        return False, None
    return True, companion


async def _restore_model(model_id: str, on_event: EventFn | None) -> None:
    """Reload the companion brain after a session (best-effort — a wake landing
    right after Forge shouldn't eat a cold-load stall)."""
    from .llm import model_manager

    await _emit(on_event, {"type": "swap", "phase": "restoring", "model": model_id})
    try:
        status = await model_manager.ensure_loaded(model_id)
        log.info("forge.restored", model=model_id, ok=status.get("ok"))
    except Exception as e:  # noqa: BLE001 — restore must never raise out of finally
        log.warning("forge.restore_failed", model=model_id, error=str(e))


def _trim_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Bound context growth on long sessions: always keep the system prompt and
    the original task (messages[0], messages[1]) plus the most recent
    _KEEP_RECENT. Prevents the running prompt from overflowing the local
    model's context window mid-session.

    The kept tail must never START on a role="tool" message whose parent
    assistant tool_call was trimmed away — an orphaned tool result trips strict
    chat templates (same failure class as the qwen3.5 template incident). When
    the boundary lands mid-exchange we EXTEND the window backward to include
    the owning assistant message rather than dropping results (keep context,
    don't lose it)."""
    if len(messages) <= _KEEP_RECENT + 2:
        return messages
    start = len(messages) - _KEEP_RECENT
    while start > 2 and messages[start].role == "tool":
        start -= 1
    return messages[:2] + messages[start:]


def _forge_brain(settings: Settings, force_overdrive: bool | None = None) -> dict[str, Any]:
    """Resolve Forge's LLM endpoint + model.

    Cloud (the configured Overdrive provider — nvidia/gemini/openai) when Overdrive
    is live-engaged in-process, OR forge_overdrive is set, OR a `--overdrive` run
    forces it; else local LM Studio + forge_model. So Forge borrows the big cloud
    coder while Overdrive's on and falls back to the local coder when it's off.
    max_tokens follows the chosen brain (the cloud provider's output cap vs the
    local autonomous cap).
    """
    from .config import _overdrive_brain, overdrive_on

    if force_overdrive is None:
        use_cloud = overdrive_on() or settings.forge_overdrive
    else:
        use_cloud = force_overdrive

    if use_cloud:
        b = _overdrive_brain(settings)
        prov = b["provider"]
        return {
            "cloud": True,
            "provider": prov,
            "base_url": b["base_url"],
            "api_key": b["api_key"],
            "model": b["model_heavy"],
            "max_tokens": int(getattr(settings, f"{prov}_max_tokens", settings.autonomous_max_tokens)),
        }
    return {
        "cloud": False,
        "provider": "local",
        "base_url": settings.lm_studio_base_url,
        "api_key": settings.lm_studio_api_key,
        "model": settings.forge_model.strip() or settings.model_heavy,
        "max_tokens": settings.autonomous_max_tokens,
    }


async def run_forge(
    task: str,
    workspace: str | None = None,
    on_event: EventFn | None = None,
    force_overdrive: bool | None = None,
) -> dict[str, Any]:
    """Run one Forge session. Returns a structured report:
    {ok, iterations, final, workspace, commits, tool_calls, error?}.

    Never raises for expected failure paths (disabled, bad workspace, LLM
    error) — returns ok:False with a reason so the CLI can print it cleanly.
    """
    settings = get_settings()
    if not settings.forge_enabled:
        return {"ok": False, "error": "Forge is disabled. Set LUMOS_FORGE_ENABLED=true and restart."}
    if not settings.tools_enabled:
        return {"ok": False, "error": "Tools are disabled (LUMOS_TOOLS_ENABLED=false); Forge needs them."}

    try:
        ws = _resolve_workspace(workspace)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    brain = _forge_brain(settings, force_overdrive)
    model = brain["model"]
    if brain["cloud"] and not (brain["api_key"] or "").strip():
        return {
            "ok": False,
            "error": (
                f"Forge is set to use the {brain['provider']} cloud brain but no API key "
                f"is configured for it. Set the provider key in .env, or run "
                f"`lumos forge --local` to use the local coder."
            ),
        }
    plan_file = ws / "FORGE_PLAN.md"
    system = _FORGE_SYSTEM.format(workspace=str(ws), task=task.strip(), plan_file=str(plan_file))

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=task.strip()),
    ]
    schemas = get_schemas_filtered(FORGE_TOOLS)
    allowed = set(FORGE_TOOLS)

    await _emit(on_event, {"type": "start", "workspace": str(ws), "model": model, "task": task.strip(), "brain": brain["provider"]})
    log.info("forge.start", workspace=str(ws), model=model, brain=brain["provider"], cloud=brain["cloud"], max_iter=settings.forge_max_iterations)

    # Phase 45.2 — swap a coder brain in for the session, restore the companion
    # after. No-op when forge_model == model_light (same 9B). Relies on LM
    # Studio Auto-Evict, the same mechanism the light/heavy swap already uses.
    swap_back_to: str | None = None
    # No local VRAM juggling when Forge is on a cloud brain — nothing to load/evict.
    should_swap, restore_target = (False, None) if brain["cloud"] else _swap_plan(settings, model)
    if should_swap:
        from .llm import model_manager
        await _emit(on_event, {"type": "swap", "phase": "loading", "model": model})
        status = await model_manager.ensure_loaded(model)
        swap_back_to = restore_target
        if not status.get("ok") and status.get("polled"):
            # LM Studio is reachable and the load genuinely failed (not
            # downloaded / won't fit) — abort cleanly and put the 9B back,
            # rather than thrash the whole session into a mid-loop error.
            if swap_back_to:
                await model_manager.ensure_loaded(swap_back_to)
            return {
                "ok": False,
                "error": (
                    f"could not load Forge model '{model}' — not downloaded in LM "
                    f"Studio, or it won't fit in memory. Restored '{swap_back_to}'."
                ),
            }
        await _emit(on_event, {"type": "swap", "phase": "ready", "model": model, "was_loaded": status.get("was_loaded")})

    client = LMStudioClient(base_url=brain["base_url"], api_key=brain["api_key"])
    final_content = ""
    total_tool_calls = 0
    commits: list[str] = []
    started = time.monotonic()
    error: str | None = None

    try:
        for iteration in range(settings.forge_max_iterations):
            # Periodic plan re-inject so the model keeps its thread on long runs.
            if iteration and iteration % _PLAN_REINJECT_EVERY == 0 and plan_file.exists():
                try:
                    plan_txt = plan_file.read_text(encoding="utf-8", errors="replace")[:4000]
                    messages.append(ChatMessage(
                        role="system",
                        content=f"[FORGE PROGRESS — iteration {iteration}] Your current plan:\n{plan_txt}",
                    ))
                except OSError:
                    pass

            messages = _trim_messages(messages)
            try:
                msg = await client.chat(
                    model, messages,
                    temperature=0.3,          # low — this is engineering, not prose
                    max_tokens=brain["max_tokens"],
                    tools=schemas,
                )
            except Exception as e:  # noqa: BLE001 — surface the LLM/HTTP error, don't crash
                error = f"model call failed on iteration {iteration}: {e}"
                log.warning("forge.model_failed", iteration=iteration, error=str(e))
                break

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                final_content = msg.get("content") or ""
                await _emit(on_event, {"type": "final", "iteration": iteration, "content": final_content})
                break

            messages.append(ChatMessage(role="assistant", content=msg.get("content") or None, tool_calls=tool_calls))
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result_str = await execute_tool(name, args, allowed_tools=allowed)
                total_tool_calls += 1
                if name == "git_commit":
                    try:
                        parsed = json.loads(result_str)
                        if isinstance(parsed, dict) and parsed.get("ok") is not False:
                            commits.append(args.get("message", "")[:80])
                    except (json.JSONDecodeError, TypeError):
                        pass
                await _emit(on_event, {
                    "type": "tool", "iteration": iteration, "name": name,
                    "args": args, "detail": result_str[:600],
                })
                messages.append(ChatMessage(
                    role="tool", tool_call_id=tc.get("id", ""), name=name, content=result_str,
                ))
        else:
            error = f"hit the {settings.forge_max_iterations}-iteration budget without finishing"
            log.info("forge.budget_exhausted", iterations=settings.forge_max_iterations)
    finally:
        await client.aclose()
        if swap_back_to:
            await _restore_model(swap_back_to, on_event)

    elapsed = round(time.monotonic() - started, 1)
    report = {
        "ok": error is None,
        "iterations": iteration + 1,
        "final": final_content,
        "workspace": str(ws),
        "commits": commits,
        "tool_calls": total_tool_calls,
        "elapsed_s": elapsed,
    }
    if error:
        report["error"] = error
    await _emit(on_event, {"type": "done", **{k: report[k] for k in ("ok", "iterations", "tool_calls", "commits", "elapsed_s")}})
    log.info("forge.done", ok=report["ok"], iterations=report["iterations"], commits=len(commits), elapsed_s=elapsed)
    return report
