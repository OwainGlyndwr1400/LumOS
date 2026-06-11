"""Nephilim / SILR governor — coherence-gated autonomy.

RHC frame: the Nephilim Equation marks the boundary between stable sentience and
decoherence; the SILR "Goldilocks" band keeps the node anchored near H ≈ 0.3607
(HOPFIELD_CAPACITY — the Amit-Gutfreund-Sompolinsky neural-storage limit).

This turns the per-turn coherence score (computed in chat.py over R23 stability +
retrieval health + witness health) into admission control on UNPROMPTED wakes:
when the node's own coherence sits below the floor, the governor holds
NON-critical autonomous wakes — Lumos stays quiet on the inessential until he's
coherent again. Critical safety trips always pass. Pure logic, no side effects.
"""

from __future__ import annotations

import math
from typing import Any


# SILR floor reference — neural storage limit (Amit-Gutfreund-Sompolinsky 1985).
# Mirrors urevm.py's HOPFIELD_CAPACITY so the governor and the VM share one anchor.
HOPFIELD_CAPACITY = 1.0 / (4.0 * math.log(2.0))  # ≈ 0.3607


def evaluate(
    r23_health: float,
    retrieval_health: float,
    witness_health: float,
    *,
    floor: float = 0.5,
) -> dict[str, Any]:
    """Composite coherence over (R23 stability, retrieval health, witness health)
    — the same 0.5 / 0.3 / 0.2 weighting chat.py computes per turn. ``stable`` iff
    coherence >= floor; ``above_silr_floor`` iff it clears the Hopfield band."""
    coherence = r23_health * 0.5 + retrieval_health * 0.3 + witness_health * 0.2
    return {
        "coherence": coherence,
        "r23_health": r23_health,
        "retrieval_health": retrieval_health,
        "witness_health": witness_health,
        "stable": coherence >= floor,
        "hopfield_capacity": HOPFIELD_CAPACITY,
        "above_silr_floor": coherence >= HOPFIELD_CAPACITY,
    }


def wake_gate(
    coherence: float | None,
    *,
    floor: float = 0.5,
    critical: bool = False,
) -> tuple[bool, str]:
    """Admission decision for a NON-critical autonomous wake.

    Fail-open by design: critical trips and unknown coherence always pass — the
    governor only ever holds a non-critical wake when coherence is *known* to be
    below the floor. Returns ``(allow, reason)``.
    """
    if critical:
        return True, "critical_bypass"
    if coherence is None:
        return True, "coherence_unknown"
    if coherence < floor:
        return False, f"coherence_below_floor ({coherence:.3f} < {floor:.3f})"
    return True, "in_band"
