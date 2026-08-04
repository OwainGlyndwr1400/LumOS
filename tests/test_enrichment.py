"""Self-updating entity memory (#3) — offline tests for the pure pieces.

No web/LLM is hit: we exercise entity extraction, queue dedup + caps, the
codex→KnowledgeChunk bridge, and the doctrine gate (disabled = worker no-ops).
"""

from pathlib import Path

from lumos_node import enrichment
from lumos_node.enrichment import (
    EntityRef,
    extract_entities,
    iter_entity_codex_chunks,
)

# ── Pure extraction: wake trips → enrichable entities ────────────────────────

def test_extract_satellite_vessel_disaster():
    trips = [
        {"kind": "satellite_overhead", "data": {"norad": "25544", "name": "ISS (ZARYA)", "country": "ISS"}},
        {"kind": "vessel", "data": {"mmsi": "212345000", "name": "NORDIC STAR"}},
        {"kind": "disaster", "data": {"event_id": "TC1024", "type_label": "Tropical cyclone", "country": "PH", "title": "Cyclone Odette"}},
    ]
    refs = extract_entities(trips)
    keys = {r.key for r in refs}
    assert keys == {"sat:25544", "vessel:212345000", "gdacs:TC1024"}
    sat = next(r for r in refs if r.kind == "satellite")
    assert "25544" in sat.query and "ISS" in sat.name


def test_extract_military_aircraft_but_not_civilian():
    # Military aircraft ARE enriched (durable, identifiable airframes, like sats);
    # civilian 'aircraft' (private/jet/commercial) are NOT.
    trips = [
        {"kind": "military_air", "data": {"hex": "43C6A1", "callsign": "RRR7229", "type_code": "A332"}},
        {"kind": "aircraft", "data": {"hex": "abc123", "callsign": "BAW14", "category": "commercial"}},
    ]
    refs = extract_entities(trips)
    assert {r.key for r in refs} == {"milair:43C6A1"}     # civilian skipped
    mil = refs[0]
    assert mil.kind == "military_aircraft"
    assert "RRR7229" in mil.query and "A332" in mil.query


def test_extract_skips_ephemeral_kinds():
    # Civilian aircraft / trains / weather are ephemeral — not worth a codex entry.
    trips = [
        {"kind": "aircraft", "data": {"hex": "abc123"}},
        {"kind": "train", "data": {"uid": "X1"}},
        {"kind": "severe_weather", "data": {}},
    ]
    assert extract_entities(trips) == []


def test_extract_dedups_within_batch():
    trips = [
        {"kind": "satellite_overhead", "data": {"norad": "25544", "name": "ISS"}},
        {"kind": "recon_satellite", "data": {"norad": "25544", "name": "ISS"}},
    ]
    refs = extract_entities(trips)
    assert len(refs) == 1


# ── Queue: dedup vs seen, cap enforcement (isolated tmp data dir) ────────────

def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(enrichment, "_data_dir", lambda s: Path(tmp_path))


def test_queue_dedups_against_seen(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    from lumos_node.config import get_settings
    s = get_settings()
    refs = [EntityRef("sat:1", "satellite", "SAT ONE", "q1")]
    assert enrichment.queue_entities(refs, s) == 1
    # Same key again → already seen → not re-queued.
    assert enrichment.queue_entities(refs, s) == 0


def test_queue_respects_max(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    from types import SimpleNamespace
    s = SimpleNamespace(enrichment_max_queue=2)
    refs = [EntityRef(f"sat:{i}", "satellite", f"S{i}", f"q{i}") for i in range(5)]
    queued = enrichment.queue_entities(refs, s)
    assert queued == 2  # capped


# ── Codex → KnowledgeChunk bridge ────────────────────────────────────────────

def test_iter_entity_codex_chunks(tmp_path):
    import orjson
    p = tmp_path / "entity_codex.jsonl"
    p.write_bytes(
        orjson.dumps({"key": "sat:25544", "kind": "satellite", "name": "ISS",
                      "summary": "The International Space Station, a crewed LEO station."}) + b"\n"
        + orjson.dumps({"key": "bad", "name": "", "summary": ""}) + b"\n"  # skipped (empty)
    )
    chunks = list(iter_entity_codex_chunks(p))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.subject == "ISS"
    assert c.agent == "entity_codex"
    assert "International Space Station" in c.text
    assert c.ping_id == "entity:sat:25544"


def test_iter_codex_missing_file_is_empty(tmp_path):
    assert list(iter_entity_codex_chunks(tmp_path / "nope.jsonl")) == []


# ── Doctrine gate: disabled worker no-ops ────────────────────────────────────

async def test_worker_noops_when_disabled(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(enrichment, "get_settings", lambda: SimpleNamespace(enrichment_enabled=False))
    # Should return immediately without scheduling any sleep/search.
    await enrichment.enrichment_worker_loop()
