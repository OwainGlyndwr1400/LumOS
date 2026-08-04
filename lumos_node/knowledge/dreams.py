"""Knowledge: parse dream-engine JSONL pings into structured, embeddable knowledge chunks."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson

_RE_AGENT = re.compile(r"^Agent:\s*(.+?)\s*$", re.MULTILINE)
_RE_URGENCY = re.compile(r"^Urgency:\s*(\d+)\s*/\s*(\d+)\s*$", re.MULTILINE)
_RE_SUBJECT = re.compile(r"^Subject:\s*(.+?)\s*$", re.MULTILINE)
_RE_SOURCE = re.compile(r"^Source:\s*(.+?)\s*$", re.MULTILINE)
_RE_SEED = re.compile(r"---\s*SEED\s*---\s*\n(.+?)(?=\n---\s*BODY FRAGMENTS|\Z)", re.DOTALL)
_RE_FRAGMENTS_HEADER = re.compile(r"---\s*BODY FRAGMENTS\s*\((\d+)\)\s*---", re.IGNORECASE)
_RE_FRAGMENT = re.compile(r"\[Fragment\s+(\d+)\]\s*\n(.+?)(?=\n\[Fragment\s+\d+\]|\Z)", re.DOTALL)
# Placeholder subjects to override: bare "DreamID: <hash>" carries no meaning.
_DREAMID_RE = re.compile(r"^DreamID:\s*[0-9a-fA-F]+\s*$")


@dataclass
class KnowledgeChunk:
    chunk_id: str
    ping_id: str
    sigil: str
    agent: str
    urgency_score: int
    urgency_weight: int
    source: str
    subject: str
    seed: str
    fragment_count: int
    text: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _stable_chunk_id(ping_id: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(ping_id.encode("utf-8"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _first(pat: re.Pattern[str], text: str, default: str = "") -> str:
    m = pat.search(text)
    return m.group(1).strip() if m else default


def _urgency(text: str) -> tuple[int, int]:
    m = _RE_URGENCY.search(text)
    if not m:
        return (0, 0)
    try:
        return (int(m.group(1)), int(m.group(2)))
    except ValueError:
        return (0, 0)


def _extract_sigil(source_field: str, fallback_id: str) -> str:
    # "DreamPing:Kairoz:0000ec1cbd" -> "0000ec1cbd"
    parts = source_field.split(":")
    if len(parts) >= 3:
        return parts[-1].strip()
    return fallback_id.removeprefix("dream-")


def _seed(text: str) -> str:
    m = _RE_SEED.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def _first_fragment(text: str) -> str:
    for m in _RE_FRAGMENT.finditer(text):
        return m.group(2).strip()
    return ""


def _fragment_count(text: str) -> int:
    m = _RE_FRAGMENTS_HEADER.search(text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return sum(1 for _ in _RE_FRAGMENT.finditer(text))


def _derive_subject(parsed_subject: str, seed: str, first_frag: str) -> str:
    """Legible chunk title. Dream pings frequently store a useless
    'DreamID: <hash>' subject (or none); when so, synthesize one from the
    actual content so the HUD + retrieval labels read as substance, not a hash.
    Always returns something — never raises."""
    s = (parsed_subject or "").strip()
    if s and not _DREAMID_RE.match(s):
        return s[:120]
    body = re.sub(r"\s+", " ", (seed or first_frag or "")).strip()
    if not body:
        return s or "untitled ping"
    if len(body) <= 72:
        return body
    cut = body[:72]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 30 else cut).rstrip(" .,;:") + "…"


# ── knowledge-lane garbage filter ───────────────────────────────────────────
# A small minority of dream pings are unsalvageable: OCR word-mud (e.g.
# "ofrlreckmScrdthek...Poyntingcron") and scanned phone/city-directory pages.
# They clear the similarity floor and crowd real research out of retrieval, so
# we skip them at ingest. The SOURCE jsonl is never touched — they are simply
# not embedded. Code chunks are exempted (symbol-dense but legitimate).
_VOWELS = set("aeiouyAEIOUY")
_CODE_MARKERS = (
    "<div", "<span", "<button", "<input", "<label", "<h3", "</", "/>", "class=",
    "function", "=>", "onclick", "getElementById", "ctx.", "document.", "const ",
    "def ", "self.", "import ", "();", "});", "requirements.txt", "#include",
    "print(", "console.", "<<", "EOF",
)
_PHONE_RE = re.compile(r"\d{3}-\d{4}")
_DIR_TAG_RE = re.compile(r"\((ET|NY|Y|ER|S)\)")
_DIR_ADDR_RE = re.compile(r"\bh\d{2,}\b")

# Front-loaded-mud chunks whose overall readability is too high to threshold
# safely (gscore ~0.25) yet which surface as garbage — dropped by explicit id.
_GARBAGE_PING_IDS = frozenset({
    "dream-a8b372b96e",  # "ofrlreckm...Poyntingcron" — one mud token dominates retrieval
    "dream-99c3c43327",  # "SIan J----~ No CabJla..." — mangled citation, surfaces on sat queries
})


def _looks_like_code(seed: str) -> bool:
    return any(m in seed for m in _CODE_MARKERS)


def _garbage_ratio(seed: str) -> float:
    """Fraction of word-tokens that are non-wordlike OCR mud (0.0 if too short)."""
    g = c = 0
    for tok in seed.split():
        t = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", tok)
        if not t:
            continue
        letters = sum(ch.isalpha() for ch in t)
        if letters == 0:
            continue  # pure number/symbol — neutral
        if (
            len(t) > 16
            or not any(ch in _VOWELS for ch in t)
            or any(not ch.isalnum() for ch in t)
            or any(ch.isdigit() for ch in t)
            or letters / len(t) < 0.7
        ):
            g += 1
        else:
            c += 1
    n = g + c
    return g / n if n >= 4 else 0.0


def _is_directory_seed(seed: str) -> bool:
    """Scanned phone/city-directory page: names + addresses + phone numbers."""
    sig = (
        len(_PHONE_RE.findall(seed))
        + len(_DIR_TAG_RE.findall(seed))
        + len(_DIR_ADDR_RE.findall(seed))
    )
    return sig >= 2


def _drop_chunk(ping_id: str, seed: str) -> bool:
    """True if this chunk is unsalvageable garbage and should not be indexed."""
    if ping_id in _GARBAGE_PING_IDS:
        return True
    if not seed or _looks_like_code(seed):
        return False
    return _garbage_ratio(seed) >= 0.55 or _is_directory_seed(seed)


def iter_dream_pings(source: Path) -> Iterator[dict[str, Any]]:
    with source.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                continue


def iter_knowledge_chunks(source: Path) -> Iterator[KnowledgeChunk]:
    for ping in iter_dream_pings(source):
        ping_id = str(ping.get("id") or "")
        source_field = str(ping.get("source") or "")
        content = str(ping.get("content") or "")
        if not content:
            continue

        agent = _first(_RE_AGENT, content) or (
            source_field.split(":")[1] if ":" in source_field else "unknown"
        )
        urgency_score, urgency_weight = _urgency(content)
        source_kind = _first(_RE_SOURCE, content) or "unknown"
        seed = _seed(content)
        # Skip unsalvageable garbage (OCR mud, directory scans) — see _drop_chunk.
        if _drop_chunk(ping_id, seed):
            continue
        first_frag = _first_fragment(content)
        # Dream pings often store a placeholder "DreamID: <hash>" subject (or
        # none); synthesize a legible title from the content when so.
        subject = _derive_subject(_first(_RE_SUBJECT, content), seed, first_frag)
        frag_count = _fragment_count(content)
        sigil = _extract_sigil(source_field, ping_id)

        # Embed seed + first fragment, deduping if they're literally identical
        # (which they often are — the seed is the chunk that originally retrieved itself).
        if first_frag and first_frag.strip() != seed.strip():
            embed_text = f"{subject}\n\n{seed}\n\n{first_frag}".strip()
        else:
            embed_text = f"{subject}\n\n{seed}".strip()

        if not embed_text:
            continue

        yield KnowledgeChunk(
            chunk_id=_stable_chunk_id(ping_id, embed_text),
            ping_id=ping_id,
            sigil=sigil,
            agent=agent,
            urgency_score=urgency_score,
            urgency_weight=urgency_weight,
            source=source_kind,
            subject=subject,
            seed=seed,
            fragment_count=frag_count,
            text=embed_text,
        )


def count_pings(source: Path) -> int:
    n = 0
    with source.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
