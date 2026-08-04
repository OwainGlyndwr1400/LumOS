"""Schumann ELF feed — pure parser + honest-null behaviour. No network."""

from lumos_node.standing_orders import compute_rhc_variables
from lumos_node.telemetry.schumann import parse_schumann


def test_parse_autodetects_common_fields():
    r = parse_schumann({"frequency": 7.83, "amplitude": 142.5})
    assert r["ok"] is True
    assert r["amplitude"] == 142.5
    assert r["frequency_hz"] == 7.83


def test_parse_explicit_dotted_key_override():
    r = parse_schumann({"data": {"amp": "88.0"}}, amp_key="data.amp")
    assert r["ok"] is True and r["amplitude"] == 88.0


def test_parse_rejects_amplitudeless_payload():
    assert parse_schumann({"foo": "bar"})["ok"] is False
    assert parse_schumann("not a dict")["ok"] is False
    assert parse_schumann(None)["ok"] is False


def test_rhc_fills_schumann_when_feed_ok():
    sch = {"ok": True, "amplitude": 142.5, "frequency_hz": 7.83,
           "source": "https://x", "confidence": "single-station-proxy"}
    ap = compute_rhc_variables({}, {}, {}, sch)["atmospheric_pressure"]
    assert ap["schumann_amplitude"] == 142.5
    assert ap["schumann_confidence"] == "single-station-proxy"
    assert "_missing" not in ap                     # term is complete now


def test_rhc_keeps_honest_null_when_no_feed():
    # No schumann arg (default None) → null + a _missing note, never faked.
    ap = compute_rhc_variables({}, {}, {})["atmospheric_pressure"]
    assert ap["schumann_amplitude"] is None
    assert "schumann" in ap["_missing"].lower()
    # A configured-but-unreachable feed carries its reason into the flag.
    ap2 = compute_rhc_variables({}, {}, {}, {"ok": False, "amplitude": None, "confidence": "fetch failed: timeout"})["atmospheric_pressure"]
    assert ap2["schumann_amplitude"] is None
    assert "timeout" in ap2["_missing"]
