"""Standing Orders (#5) — offline tests for the two security locks + matching.

The whole safety model is (1) HMAC signature authenticity and (2) the action
whitelist. Both are hammered here, plus trigger matching and the disabled gate.
No network / LLM / Discord is touched.
"""


from lumos_node import standing_orders as so
from lumos_node.standing_orders import (
    ALLOWED_ACTIONS,
    compute_rhc_variables,
    compute_signature,
    order_matches,
    parse_xray_flux,
    spring_tide_factor,
    verify_playbook,
)

_TOKEN = "test-secret-token"


def _signed(orders):
    return {"orders": orders, "signature": compute_signature(orders, _TOKEN)}


# ── Lock 1: signature authenticity ───────────────────────────────────────────

def test_valid_signature_accepted():
    pb = _signed([{"name": "a", "when": {"kind": "disaster"}, "do": []}])
    ok, _ = verify_playbook(pb, _TOKEN)
    assert ok is True


def test_tampered_orders_rejected():
    # Sign one set, then mutate the orders without re-signing → must be refused.
    pb = _signed([{"name": "a", "when": {"kind": "disaster"}, "do": []}])
    pb["orders"].append({"name": "evil", "when": {"kind": "vessel"}, "do": [{"action": "discord_dm"}]})
    ok, reason = verify_playbook(pb, _TOKEN)
    assert ok is False
    assert "signature" in reason.lower()


def test_wrong_token_rejected():
    pb = _signed([{"name": "a", "when": {"kind": "disaster"}, "do": []}])
    ok, _ = verify_playbook(pb, "different-token")
    assert ok is False


def test_missing_token_refuses_everything():
    pb = _signed([{"name": "a", "when": {"kind": "disaster"}, "do": []}])
    ok, reason = verify_playbook(pb, "")
    assert ok is False and "token" in reason.lower()


def test_unsigned_playbook_rejected():
    ok, reason = verify_playbook({"orders": []}, _TOKEN)
    assert ok is False and "unsigned" in reason.lower()


# ── Lock 2: action whitelist ─────────────────────────────────────────────────

def test_whitelist_is_exactly_the_expected_set():
    assert {
        "dossier", "note", "briefing", "discord_dm", "workspace_file", "rhc_snapshot",
    } == ALLOWED_ACTIONS


async def test_unknown_action_refused_even_when_signed(tmp_path, monkeypatch):
    # A validly-signed order containing a bogus action → that step is REFUSED,
    # not executed. (Signature proves authorship, whitelist bounds capability.)
    from types import SimpleNamespace
    s = SimpleNamespace(operator_lat=51.6, operator_lon=-4.0)
    order = {"name": "x", "when": {"kind": "disaster"}, "do": [{"action": "run_shell", "cmd": "rm -rf /"}]}
    results = await so.execute_order(s, order, {"kind": "disaster", "data": {}})
    assert results == [{"action": "run_shell", "result": "REFUSED (not in whitelist)"}]


# ── Trigger matching ─────────────────────────────────────────────────────────

def test_match_on_kind():
    o = {"when": {"kind": "regulus_lock"}}
    assert order_matches(o, {"kind": "regulus_lock", "data": {}}) is True
    assert order_matches(o, {"kind": "vessel", "data": {}}) is False


def test_match_extra_condition_substring():
    o = {"when": {"kind": "disaster", "level": "red"}}
    assert order_matches(o, {"kind": "disaster", "data": {"level": "red"}}) is True
    assert order_matches(o, {"kind": "disaster", "data": {"level": "orange"}}) is False


def test_malformed_when_never_matches():
    # Fail-closed: no 'kind' → matches nothing (won't fire on everything).
    assert order_matches({"when": {}}, {"kind": "disaster", "data": {}}) is False
    assert order_matches({}, {"kind": "disaster", "data": {}}) is False


def test_flare_and_quake_kinds_match():
    # Lock the cosmic kind strings the flare/quake orders trigger on — if a
    # refactor renames these, this test catches it before the orders go silent.
    assert order_matches({"when": {"kind": "solar_flare"}},
                         {"kind": "solar_flare", "data": {"magnitude_text": "M5.2"}})
    assert order_matches({"when": {"kind": "major_earthquake"}},
                         {"kind": "major_earthquake", "data": {"magnitude": 6.8}})
    # Optional X-class-only restriction via substring on magnitude_text.
    x_only = {"when": {"kind": "solar_flare", "magnitude_text": "X"}}
    assert order_matches(x_only, {"kind": "solar_flare", "data": {"magnitude_text": "X1.0"}})
    assert not order_matches(x_only, {"kind": "solar_flare", "data": {"magnitude_text": "M5.2"}})


# ── Workspace path-traversal guard ───────────────────────────────────────────

def test_workspace_target_rejects_traversal(monkeypatch):
    # Patch the git validator to accept cwd as a repo, then a path inside is
    # allowed but a '../' traversal outside the workspace is refused.
    from pathlib import Path
    from types import SimpleNamespace

    import lumos_node.tools.git_tools as gt
    repo = Path.cwd()
    monkeypatch.setattr(gt, "_check_repo_path", lambda w: repo)
    s = SimpleNamespace()
    assert so._safe_workspace_target(s, str(repo), "notes/ok.md") is not None
    assert so._safe_workspace_target(s, str(repo), "../../escape.md") is None


# ── RHC seismic variables (the analysis dataset) ─────────────────────────────

def test_parse_xray_flux():
    assert parse_xray_flux("M2.3") == 2.3e-5
    assert parse_xray_flux("X1.0") == 1e-4
    assert parse_xray_flux("C5") == 5e-6
    assert parse_xray_flux("M") == 1e-5      # bare letter → ×1.0
    assert parse_xray_flux(None) is None
    assert parse_xray_flux("Z9") is None     # unknown class letter


def test_spring_tide_factor():
    assert spring_tide_factor(0) == 1.0      # new moon → spring tide
    assert spring_tide_factor(100) == 1.0    # full moon → spring tide
    assert spring_tide_factor(50) == 0.0     # quarter → neap
    assert spring_tide_factor(None) is None


def test_compute_rhc_full_payload():
    snap = {
        "solar_wind": {"speed_kms": 532, "bz_nt": -8.0},
        "xray": {"current_class": "M2.3"},
        "geomagnetic": {"kp": 3.33},
        "earthquakes_recent": [{"magnitude": 5.1}, {"magnitude": 4.2}],
    }
    cycle = {"current_f107": 155, "current_ssn": 42}
    grid = {"moon": {"illumination_percent": 90, "phase_name": "Waxing Gibbous", "zodiac_sign": "Leo"},
            "fixed_stars": {"Regulus": {"alt_deg": 22.5, "az_deg": 90.1}}}
    r = compute_rhc_variables(snap, cycle, grid)
    assert r["cme_intensity"]["xray_flux_wm2"] == 2.3e-5
    assert r["cme_intensity"]["bz_south_coupling"] == 8.0     # southward Bz counted
    assert r["cme_intensity"]["index"] is not None
    assert r["atmospheric_pressure"]["ionospheric_index"] == 197.0   # f107 + ssn
    assert r["crustal_stress"]["spring_tide_factor"] == 0.8          # near-full
    assert r["crustal_stress"]["recent_quake_max_mag"] == 5.1
    assert r["kp"] == 3.33


def test_compute_rhc_honest_nulls_when_feeds_missing():
    # The two unmeasured sub-variables must be null + flagged, never faked; and a
    # totally empty payload must not crash.
    r = compute_rhc_variables({}, {}, {})
    assert r["atmospheric_pressure"]["schumann_amplitude"] is None
    assert "schumann" in r["atmospheric_pressure"]["_missing"].lower()
    assert r["crustal_stress"]["tectonic_strain"] is None
    assert "strain" in r["crustal_stress"]["_missing"].lower()
    assert r["cme_intensity"]["index"] is None       # no data → no fabricated index
    # northward Bz → coupling 0 (matches the model: northward = weak coupling)
    r2 = compute_rhc_variables({"solar_wind": {"speed_kms": 400, "bz_nt": 5.0}, "xray": {"current_class": "C1.0"}}, {}, {})
    assert r2["cme_intensity"]["bz_south_coupling"] == 0.0


# ── Disabled gate ────────────────────────────────────────────────────────────

async def test_fire_orders_noop_when_disabled():
    from types import SimpleNamespace
    n = await so.fire_orders([{"kind": "disaster", "data": {}}], SimpleNamespace(standing_orders_enabled=False))
    assert n == 0
