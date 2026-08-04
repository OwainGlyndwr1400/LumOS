"""Deterministic Enochian topology overlays for Lumos memory and telemetry.

These helpers turn the symbolic layer into stable metadata only. They do not
change vector contents, source JSONL files, FAISS indexes, or retrieval scores.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

LOAGAETH_SIZE = 49

LOAGAETH_DOMAINS: tuple[str, ...] = (
    "astronomy",
    "resonance",
    "myth",
    "code",
    "memory",
    "gnosis",
    "geometry",
    "language",
    "history",
    "archaeology",
    "physics",
    "consciousness",
    "cosmology",
    "music",
    "ritual",
    "alchemy",
    "botany",
    "medicine",
    "weather",
    "seismology",
    "space_weather",
    "near_earth_objects",
    "airspace",
    "maritime",
    "rail",
    "geography",
    "mathematics",
    "logic",
    "quaternion",
    "scalar_field",
    "sacred_text",
    "hermeticism",
    "gnostic_text",
    "celtic",
    "welsh",
    "tesla",
    "davinci",
    "newton",
    "voynich",
    "sphinx",
    "regulus",
    "gobekli_tepe",
    "stonehenge",
    "serpent_mound",
    "newgrange",
    "antarctica",
    "operator",
    "system",
    "unknown",
)

DOMAIN_KEYWORDS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, ("star", "planet", "azimuth", "altitude", "eclipse", "sidereal")),
    (2, ("frequency", "hz", "resonance", "cymatic", "harmonic")),
    (3, ("myth", "archetype", "lion", "sophia", "demiurge")),
    (4, ("python", "typescript", "fastapi", "react", "code", "script")),
    (7, ("geometry", "heptagon", "sigil", "triangle", "angle")),
    (10, ("gobekli", "sphinx", "stonehenge", "mound", "newgrange")),
    (11, ("quantum", "physics", "mass gap", "field", "particle")),
    (12, ("consciousness", "observer", "mind", "awareness")),
    (20, ("quake", "earthquake", "seismic", "usgs")),
    (21, ("solar", "kp", "flare", "geomagnetic", "bz", "solar wind")),
    (22, ("neo", "asteroid", "near-earth", "lunar distance")),
    (23, ("aircraft", "airspace", "opensky", "adsb", "callsign")),
    (24, ("ship", "vessel", "maritime", "ais", "naval")),
    (27, ("math", "theorem", "proof", "equation", "prime")),
    (29, ("quaternion", "triskelion", "ure-vm", "r23")),
    (31, ("enoch", "nag hammadi", "dead sea", "raziel", "hermetic corpus")),
    (39, ("voynich", "cipher", "manuscript")),
    (40, ("sphinx", "giza")),
    (41, ("regulus", "lion gate")),
    (47, ("erydir", "operator", "ceisiwr")),
    (48, ("lumos", "node", "system", "hud")),
)

SENTINEL_BY_KIND: dict[str, tuple[str, str]] = {
    "geomagnetic_storm": ("BOBOGEL", "BOBOGEL_THRESHOLD_BREACH"),
    "solar_flare": ("BOBOGEL", "BOBOGEL_THRESHOLD_BREACH"),
    "solar_wind_high": ("BOBOGEL", "BOBOGEL_THRESHOLD_BREACH"),
    "bz_southward": ("BOBOGEL", "BOBOGEL_THRESHOLD_BREACH"),
    "near_earth_pass": ("BOBOGEL", "BOBOGEL_THRESHOLD_BREACH"),
    "major_earthquake": ("BYNEPOR", "BYNEPOR_THRESHOLD_BREACH"),
    "natural_events_surge": ("BYNEPOR", "BYNEPOR_THRESHOLD_BREACH"),
    "vessel": ("BABALEL", "BABALEL_THRESHOLD_BREACH"),
    "train": ("HAGONEL", "HAGONEL_THRESHOLD_BREACH"),
    "aircraft": ("HAGONEL", "HAGONEL_THRESHOLD_BREACH"),
    "military_air": ("HAGONEL", "HAGONEL_SECURITY_BREACH"),
    "gps_jamming": ("HAGONEL", "HAGONEL_SECURITY_BREACH"),
    "recon_satellite": ("BOBOGEL", "BOBOGEL_THRESHOLD_BREACH"),
    "severe_weather": ("BYNEPOR", "BYNEPOR_THRESHOLD_BREACH"),
    "regulus_rise": ("CARMARA", "CARMARA_ALIGNMENT_RISE"),
    # Eastern-gate upgrade (2026-07-15): gate entry + the due-east lock each
    # carry their own sigil — the lock IS the Sphinx alignment moment.
    "regulus_gate": ("CARMARA", "CARMARA_ALIGNMENT_GATE"),
    "regulus_lock": ("CARMARA", "CARMARA_LION_LOCK"),
}


def _stable_seed(parts: list[str]) -> bytes:
    raw = "\n".join(p for p in parts if p).encode("utf-8", errors="ignore")
    if not raw:
        raw = b"lumos"
    return hashlib.sha256(raw).digest()


def _metadata_text(metadata: dict[str, Any]) -> str:
    fields = (
        "chunk_id",
        "conversation_title",
        "subject",
        "source",
        "agent",
        "sigil",
        "text",
    )
    return " ".join(str(metadata.get(k) or "") for k in fields).lower()


def _keyword_leaf(text: str) -> int | None:
    compact = re.sub(r"\s+", " ", text.lower())
    for leaf, keywords in DOMAIN_KEYWORDS:
        if any(k in compact for k in keywords):
            return leaf
    return None


def loagaeth_coordinate(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic 49x49 coordinate overlay for a memory chunk."""

    text = _metadata_text(metadata)
    digest = _stable_seed([text])
    leaf = _keyword_leaf(text) or (digest[0] % LOAGAETH_SIZE) + 1
    row = (digest[1] % LOAGAETH_SIZE) + 1
    column = (digest[2] % LOAGAETH_SIZE) + 1
    return {
        "leaf": leaf,
        "row": row,
        "column": column,
        "domain": LOAGAETH_DOMAINS[leaf - 1],
    }


def enrich_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    out = dict(metadata)
    out.setdefault("loagaeth", loagaeth_coordinate(out))
    return out


def sigillum_geometry(cluster: dict[str, Any], index: int) -> dict[str, Any]:
    """Map an atlas cluster to a fixed Sigillum-style 3D position."""

    lane = str(cluster.get("lane") or "")
    cluster_id = str(cluster.get("id") or index)
    digest = _stable_seed([cluster_id, lane, str(cluster.get("label") or "")])
    if lane == "identity":
        sector = (digest[0] % 7) + 1
        sides = 7
        radius = 42.0
        ring = "heptagon"
    else:
        sector = (digest[0] % 40) + 1
        sides = 40
        radius = 92.0
        ring = "outer40"
    theta = ((sector - 1) / sides) * math.tau
    triad = (sector - 1) % 3
    z = (triad - 1) * 18.0
    return {
        "ring": ring,
        "sector": sector,
        "theta_deg": round(math.degrees(theta), 3),
        "radius": radius,
        "x": round(math.cos(theta) * radius, 6),
        "y": round(math.sin(theta) * radius, 6),
        "z": round(z, 6),
        "triad": triad,
    }


def apply_sigillum_geometry(atlas: dict[str, Any]) -> dict[str, Any]:
    clusters = atlas.get("clusters")
    if not isinstance(clusters, list):
        return atlas
    enriched = []
    for index, raw in enumerate(clusters):
        if not isinstance(raw, dict):
            enriched.append(raw)
            continue
        cluster = dict(raw)
        cluster.setdefault("sigillum", sigillum_geometry(cluster, index))
        enriched.append(cluster)
    return {**atlas, "clusters": enriched, "sigillum": {"inner": 7, "outer": 40}}


def sentinel_for_kind(kind: str) -> dict[str, str]:
    sentinel, code = SENTINEL_BY_KIND.get(
        kind,
        ("HAGONEL", f"HAGONEL_{re.sub(r'[^A-Z0-9]+', '_', kind.upper()).strip('_') or 'EVENT'}"),
    )
    return {"sentinel": sentinel, "system_wake_code": code}


def enrich_event(event: dict[str, Any]) -> dict[str, Any]:
    kind = str(event.get("kind") or "")
    return {**event, **sentinel_for_kind(kind)}
