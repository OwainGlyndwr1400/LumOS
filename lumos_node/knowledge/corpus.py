"""Corpus ingestion — research files (CSV / MD / TXT) → knowledge chunks.

Second source for the KNOWLEDGE lane alongside dream pings. The operator's
research corpus (RHC theorem tables, opcode breakdowns, blueprints) lives as
files; pointing LUMOS_KNOWLEDGE_EXTRA_DIR at the folder folds every row /
paragraph into the same FAISS lane the retrieval pipeline already injects —
so equations reach the model through the EXISTING 6 knowledge slots at zero
added context cost, instead of via per-turn file-tool reads.

CSV: one chunk per data row ("Header: value" lines — theorem rows are
naturally chunk-sized). MD/TXT: paragraph-accumulated chunks. All emitted as
KnowledgeChunk so the composer + HUD render them exactly like dream pings:
subject = row title, agent = "rhc_corpus", source = filename.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

from .dreams import KnowledgeChunk


_CSV_MAX_CHARS = 1500     # one theorem row, fully stated
_TEXT_MAX_CHARS = 1200    # matches the retrieval chunk-budget scale
_SUPPORTED = {".csv", ".md", ".txt"}


def _chunk_id(source_name: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(source_name.encode("utf-8"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _make_chunk(source_name: str, subject: str, text: str, seed: str = "") -> KnowledgeChunk:
    cid = _chunk_id(source_name, text)
    return KnowledgeChunk(
        chunk_id=cid,
        ping_id=f"corpus-{cid}",
        sigil=cid[:10],
        agent="rhc_corpus",
        urgency_score=0,
        urgency_weight=0,
        source=source_name,
        subject=subject[:160],
        seed=seed[:400],
        fragment_count=1,
        text=text,
    )


def _iter_csv_chunks(path: Path) -> Iterator[KnowledgeChunk]:
    """One chunk per data row: 'Header: value' lines for every non-empty cell.
    The first non-empty cell doubles as the subject (theorem/concept name)."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            # Some exports are tab-delimited despite the .csv extension —
            # sniff, falling back to comma.
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError):
        return
    if len(rows) < 2:
        return
    headers = [h.strip() for h in rows[0]]
    for row in rows[1:]:
        cells = [(headers[i] if i < len(headers) else f"col{i}", (c or "").strip())
                 for i, c in enumerate(row)]
        cells = [(h, v) for h, v in cells if v]
        if not cells:
            continue
        subject = cells[0][1]
        # The equation column (when present) doubles as the seed so the
        # composer's compact rendering leads with the formula.
        seed = next((v for h, v in cells if "equation" in h.lower() or "formula" in h.lower()), "")
        body = "\n".join(f"{h}: {v}" for h, v in cells)
        text = body[:_CSV_MAX_CHARS]
        if len(text) < 20:  # header junk / separator rows
            continue
        yield _make_chunk(path.name, subject, text, seed)


def _iter_text_chunks(path: Path) -> Iterator[KnowledgeChunk]:
    """Paragraph-accumulated chunks for prose files (blueprints, notes)."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    buf: list[str] = []
    size = 0
    idx = 0
    title = path.stem
    for p in paragraphs:
        if size + len(p) > _TEXT_MAX_CHARS and buf:
            idx += 1
            text = "\n\n".join(buf)
            yield _make_chunk(path.name, f"{title} §{idx}", text, buf[0][:200])
            buf, size = [], 0
        buf.append(p)
        size += len(p)
    if buf:
        idx += 1
        text = "\n\n".join(buf)
        yield _make_chunk(path.name, f"{title} §{idx}", text, buf[0][:200])


def corpus_files(extra_dir: Path) -> list[Path]:
    """Supported research files under extra_dir, stable order."""
    if not extra_dir.is_dir():
        return []
    return sorted(
        p for p in extra_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED
    )


def iter_corpus_chunks(extra_dir: Path) -> Iterator[KnowledgeChunk]:
    for path in corpus_files(extra_dir):
        if path.suffix.lower() == ".csv":
            yield from _iter_csv_chunks(path)
        else:
            yield from _iter_text_chunks(path)


def corpus_signature(extra_dir: Path) -> tuple[int, float]:
    """Aggregate (total_size, max_mtime) so the knowledge manifest can detect
    corpus changes and trigger a rebuild — mirrors _source_signature."""
    files = corpus_files(extra_dir)
    if not files:
        return (0, 0.0)
    total = 0
    newest = 0.0
    for p in files:
        st = p.stat()
        total += st.st_size
        newest = max(newest, st.st_mtime)
    return (total, newest)
