"""Triskelion routing — pure-function verification (Phase 43 Group B #4).

The routing function is pure (no VM/Settings/network), so the corpus status→action
table and all safety invariants are tested in isolation. The headline guarantee:
with the master flag OFF the decision is EXACTLY neutral (turn byte-identical to today),
and the destructive flags (requery/abort) NEVER flip without the hard gate.
"""

import lumos_node.triskelion_routing as tr


def _settings(routing=False, hard=False):
    return {"routing_enabled": routing, "hard_gate_enabled": hard}


# ── master flag OFF → neutral ─────────────────────────────────────────────────

def test_triskelion_route_disabled_returns_neutral():
    # Even with the worst-case lock + forbidden, routing OFF is exactly neutral.
    d = tr.triskelion_route(
        {"status": "weak"}, {"near_forbidden": True}, _settings(routing=False, hard=True)
    )
    assert d == {
        "temperature_mult": 1.0,
        "prompt_nudge": None,
        "requery": False,
        "abort": False,
        "state": "disabled",
        "reason": "routing_disabled",
    }


# ── the corpus table (routing ON) ─────────────────────────────────────────────

def test_triskelion_route_moderate_drops_temperature():
    d = tr.triskelion_route({"status": "moderate"}, {}, _settings(routing=True))
    assert d["temperature_mult"] == 0.85
    assert d["prompt_nudge"] is None
    assert d["requery"] is False and d["abort"] is False
    assert d["state"] == "moderate"


def test_triskelion_route_strong_proceeds():
    d = tr.triskelion_route({"status": "strong"}, {}, _settings(routing=True))
    assert 0.7 <= d["temperature_mult"] <= 1.2
    assert d["temperature_mult"] == 0.98
    assert d["prompt_nudge"] is None and not d["requery"] and not d["abort"]
    assert d["state"] == "strong"


def test_triskelion_route_weak_nudges_not_requery_when_gate_off():
    d = tr.triskelion_route({"status": "weak"}, {}, _settings(routing=True, hard=False))
    assert isinstance(d["prompt_nudge"], str) and d["prompt_nudge"]
    assert d["requery"] is False and d["abort"] is False
    assert d["temperature_mult"] == 1.0
    assert d["state"] == "weak" and d["reason"] == "weak_nudge"


def test_triskelion_route_weak_requery_when_gate_on():
    d = tr.triskelion_route({"status": "weak"}, {}, _settings(routing=True, hard=True))
    assert d["requery"] is True
    assert d["reason"] == "weak_requery"


def test_triskelion_route_forbidden_telemetry_only_when_gate_off():
    d = tr.triskelion_route({"status": "strong"}, {"near_forbidden": True}, _settings(routing=True))
    assert d["abort"] is False
    assert d["state"] == "forbidden" and d["reason"] == "forbidden_telemetry_only"
    assert d["temperature_mult"] == 1.0 and d["prompt_nudge"] is None


def test_triskelion_route_forbidden_aborts_when_gate_on():
    d = tr.triskelion_route(
        {"status": "strong"}, {"near_forbidden": True}, _settings(routing=True, hard=True)
    )
    assert d["abort"] is True
    assert d["reason"] == "forbidden_abort"


def test_triskelion_route_forbidden_takes_precedence_over_weak():
    d = tr.triskelion_route({"status": "weak"}, {"near_forbidden": True}, _settings(routing=True))
    assert d["state"] == "forbidden"   # forbidden branch wins


# ── safety invariants ─────────────────────────────────────────────────────────

def test_triskelion_route_temperature_always_bounded():
    for status in ("strong", "moderate", "weak", "bogus", None):
        for nf in (True, False):
            for hard in (True, False):
                d = tr.triskelion_route(
                    {"status": status}, {"near_forbidden": nf}, _settings(routing=True, hard=hard)
                )
                assert 0.7 <= d["temperature_mult"] <= 1.2, (status, nf, hard, d)


def test_triskelion_route_unknown_status_safe_default():
    d = tr.triskelion_route({"status": "garbage"}, {}, _settings(routing=True))
    assert d["temperature_mult"] == 0.85
    assert d["state"] == "moderate" and d["reason"] == "unknown_status_safe_default"
    assert not d["requery"] and not d["abort"]


def test_triskelion_route_no_destructive_flags_without_hard_gate():
    for status in ("strong", "moderate", "weak", "bogus"):
        for nf in (True, False):
            d = tr.triskelion_route(
                {"status": status}, {"near_forbidden": nf}, _settings(routing=True, hard=False)
            )
            assert d["requery"] is False and d["abort"] is False, (status, nf, d)


def test_triskelion_route_is_pure_and_handles_none():
    a = tr.triskelion_route({"status": "moderate"}, {}, _settings(routing=True))
    b = tr.triskelion_route({"status": "moderate"}, {}, _settings(routing=True))
    assert a == b                                   # deterministic
    # None inputs must not raise (routing off → neutral)
    assert tr.triskelion_route(None, None, None)["state"] == "disabled"


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
