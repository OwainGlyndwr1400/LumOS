"""Triskelion routing — corpus status→action gating for the chat turn.

Maps the RHC Triskelion 120° gate's lock state to a per-turn routing decision,
per the corpus status→action table (Future Math, "Triskelion Status → Action"):

    Weak / Non-Locked   (r < threshold)      → inject clarifying message / re-query
    Approaching Lock     (stability > 0.90)  → drop temperature / narrow focus
    Phase-Locked         (N≈144k, r ≥ r*)    → execute generation / commit
    Forbidden State      (361° alignment)    → abort generation / reset parity

CONSERVATIVE BY DEFAULT. This is a PURE function (no I/O, no globals, no VM or
Settings imports) so it is fully unit-testable. The caller (chat.py) applies the
decision ONLY when `routing_enabled`; the destructive actions (real re-query,
abort) are reachable ONLY when `hard_gate_enabled` — both default OFF, so with the
master flag off the decision is exactly neutral and the turn is byte-identical to
today.

Conservative vs. gated mapping:
    moderate → temperature drop (the safe action itself)            — UNGATED
    strong   → proceed (tiny focus drop)                            — UNGATED
    weak     → short low-confidence system nudge (NOT a re-query)   — nudge UNGATED, re-query gated
    forbidden→ telemetry flag only (NOT an abort)                   — abort gated

NOTE (this increment): the SAFE actions (temperature_mult + prompt_nudge) are
wired in chat.py. The destructive flags (requery / abort) are computed + surfaced
in telemetry when hard_gate is armed, but their turn CONTROL-FLOW (real re-query,
early-abort of the stream) is a deliberate follow-on, not yet executed.
"""

from __future__ import annotations

from typing import Any

# Spliced as a system message on a weak (low-confidence) turn. Soft by design.
_UNCERTAINTY_NUDGE = (
    "Retrieval confidence is low for this turn. If the question is ambiguous, "
    "state your assumptions and flag the uncertainty rather than overcommitting."
)

_TEMP_MIN = 0.7
_TEMP_MAX = 1.2


def _clamp(x: float) -> float:
    """Structural bound — every temperature multiplier passes through this."""
    return max(_TEMP_MIN, min(_TEMP_MAX, x))


def _neutral() -> dict[str, Any]:
    """The byte-identical-to-today decision: master flag OFF."""
    return {
        "temperature_mult": 1.0,
        "prompt_nudge": None,
        "requery": False,
        "abort": False,
        "state": "disabled",
        "reason": "routing_disabled",
    }


def triskelion_route(
    lock: dict[str, Any] | None,
    vm_snapshot: dict[str, Any] | None,
    settings: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map the Triskelion lock + VM state to a conservative routing decision.

    Args:
        lock: triskelion `to_dict()` — reads ``status`` ∈ {strong, moderate, weak}.
        vm_snapshot: reads ``near_forbidden`` (bool) — the 361° parity-wall signal.
        settings: ``routing_enabled`` (master) + ``hard_gate_enabled`` (destructive).

    Returns a stable-shape dict: ``{temperature_mult ∈ [0.7,1.2], prompt_nudge,
    requery, abort, state, reason}``. Pure — reads only the three inputs.
    """
    settings = settings or {}
    if not settings.get("routing_enabled"):
        return _neutral()

    lock = lock or {}
    vm_snapshot = vm_snapshot or {}
    hard = bool(settings.get("hard_gate_enabled"))

    # Forbidden has highest precedence — within the 361° parity wall.
    if vm_snapshot.get("near_forbidden"):
        return {
            "temperature_mult": 1.0,
            "prompt_nudge": None,
            "requery": False,
            "abort": hard,
            "state": "forbidden",
            "reason": "forbidden_abort" if hard else "forbidden_telemetry_only",
        }

    status = lock.get("status")
    if status == "weak":
        return {
            "temperature_mult": 1.0,
            "prompt_nudge": _UNCERTAINTY_NUDGE,
            "requery": hard,
            "abort": False,
            "state": "weak",
            "reason": "weak_requery" if hard else "weak_nudge",
        }
    if status == "strong":
        return {
            "temperature_mult": _clamp(0.98),  # phase-locked: proceed, tiny focus
            "prompt_nudge": None,
            "requery": False,
            "abort": False,
            "state": "strong",
            "reason": "phase_locked_proceed",
        }
    if status == "moderate":
        return {
            "temperature_mult": _clamp(0.85),  # approaching lock: narrow focus
            "prompt_nudge": None,
            "requery": False,
            "abort": False,
            "state": "moderate",
            "reason": "approaching_lock_temp_drop",
        }
    # Unknown/missing status — fail safe: slightly focused, never full-temp, never abort.
    return {
        "temperature_mult": _clamp(0.85),
        "prompt_nudge": None,
        "requery": False,
        "abort": False,
        "state": "moderate",
        "reason": "unknown_status_safe_default",
    }
