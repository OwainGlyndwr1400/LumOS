from __future__ import annotations

from lumos_node.enochian import (
    apply_sigillum_geometry,
    enrich_event,
    loagaeth_coordinate,
)


def test_loagaeth_coordinate_is_stable_and_bounded() -> None:
    meta = {
        "chunk_id": "abc123",
        "subject": "Regulus harmonic astronomy",
        "text": "Sphinx and Regulus alignment at azimuth 90.",
    }

    first = loagaeth_coordinate(meta)
    second = loagaeth_coordinate(meta)

    assert first == second
    assert 1 <= first["leaf"] <= 49
    assert 1 <= first["row"] <= 49
    assert 1 <= first["column"] <= 49
    assert first["domain"] == "astronomy"


def test_sigillum_geometry_is_added_to_existing_atlas_payload() -> None:
    atlas = {
        "clusters": [
            {"id": "i_001", "lane": "identity", "label": "core", "size": 3},
            {"id": "k_001", "lane": "knowledge", "label": "research", "size": 5},
        ]
    }

    enriched = apply_sigillum_geometry(atlas)

    assert enriched["sigillum"] == {"inner": 7, "outer": 40}
    assert enriched["clusters"][0]["sigillum"]["ring"] == "heptagon"
    assert enriched["clusters"][1]["sigillum"]["ring"] == "outer40"


def test_sentinel_enrichment_adds_stable_wake_code() -> None:
    event = enrich_event({"kind": "major_earthquake", "description": "M6.1"})

    assert event["sentinel"] == "BYNEPOR"
    assert event["system_wake_code"] == "BYNEPOR_THRESHOLD_BREACH"
