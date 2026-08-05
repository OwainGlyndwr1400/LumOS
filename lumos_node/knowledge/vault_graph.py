"""Obsidian vault knowledge-graph — the associative third memory.

Sits ALONGSIDE the dual-lane FAISS (identity + knowledge), not replacing it.
Where FAISS finds notes by *semantic similarity* (a cosine guess at what's
related), this walks the `[[wikilink]]` edges the operator actually authored —
the spider graph you see in Obsidian's graph view. Those edges are deliberate
associations, so traversal follows real trains of thought rather than
embedding-space neighbours.

Crucially it needs NO embedding: a `[[link]]` is just text, so a regex over the
markdown rebuilds the entire graph in seconds and caches it to a tiny JSON
(names + edges only, never content). Rebuild is triggered only when files change
(cheap stat-signature) or on an explicit reindex.

Three moves it gives Lumos:
  * search(query)         — grep to find entry notes (which dot to land on)
  * read_note(name)       — a note's text + its [[links]] + backlinks (the dot's
                            content and every dot it's wired to)
  * neighbors(name, depth)— walk the graph N hops (go deeper, follow the threads)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from ..log import get_logger

log = get_logger(__name__)

# `[[Target]]`, `[[Target|alias]]`, `[[Target#heading]]`, `![[embed]]` — the
# inner capture is the same for all; we normalise alias/heading/path off below.
_WIKILINK = re.compile(r"\[\[([^\[\]]+?)\]\]")

_READ_MAX_CHARS = 8000       # cap a single note so one huge file can't flood context
_NEIGHBOR_LIMIT = 40         # max nodes returned from a traversal
_CACHE_FILE = "vault_graph.json"

# In-process singleton — built once, reused across tool calls for the lifetime
# of `lumos serve`. A reindex (tool/CLI) or a changed stat-signature rebuilds it.
_cached: VaultGraph | None = None


@dataclass
class NoteRef:
    name: str    # display stem (original filename without .md)
    vault: str   # which vault folder it came from
    path: str    # absolute path


@dataclass
class VaultGraph:
    # normalised-name → refs (a list because the same note name can exist in
    # more than one vault; dangling link targets have an edge but no ref).
    nodes: dict[str, list[NoteRef]] = field(default_factory=dict)
    outgoing: dict[str, list[str]] = field(default_factory=dict)   # name → linked names
    backlinks: dict[str, list[str]] = field(default_factory=dict)  # name → linking names
    roots: list[str] = field(default_factory=list)                 # vault dir paths
    signature: list = field(default_factory=list)                  # [file_count, bytes, max_mtime]

    # ── traversal / lookup ──────────────────────────────────────────────────

    def read_note(self, name: str, max_chars: int = _READ_MAX_CHARS) -> dict | None:
        key = _norm(name)
        refs = self.nodes.get(key)
        outgoing = self.outgoing.get(key, [])
        backlinks = self.backlinks.get(key, [])
        if not refs:
            # A concept referenced by `[[...]]` but with no file of its own —
            # Obsidian's "unresolved" node. It's still a real graph position.
            if backlinks:
                return {
                    "note": name, "exists": False, "text": "",
                    "outgoing": [], "backlinks": backlinks,
                    "hint": "no note file — a concept other notes link TO",
                }
            return None
        ref = refs[0]
        try:
            text = Path(ref.path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            text = f"[could not read {ref.path}: {e}]"
        truncated = len(text) > max_chars
        return {
            "note": ref.name, "vault": ref.vault, "path": ref.path, "exists": True,
            "text": text[:max_chars], "truncated": truncated,
            "outgoing": outgoing, "backlinks": backlinks,
            "also_in": [r.vault for r in refs[1:]],
        }

    def neighbors(self, name: str, depth: int = 1, limit: int = _NEIGHBOR_LIMIT) -> dict:
        start = _norm(name)
        seen = {start}
        frontier = [start]
        edges: list[tuple[str, str]] = []
        for _ in range(max(1, depth)):
            nxt: list[str] = []
            for node in frontier:
                for tgt in self.outgoing.get(node, []):
                    edges.append((node, tgt))
                    if tgt not in seen:
                        seen.add(tgt)
                        nxt.append(tgt)
                for src in self.backlinks.get(node, []):
                    edges.append((src, node))
                    if src not in seen:
                        seen.add(src)
                        nxt.append(src)
            frontier = nxt
            if len(seen) >= limit or not frontier:
                break
        connected = [n for n in seen if n != start]
        return {
            "note": self._display(start),
            "depth": depth,
            "connected_count": len(connected),
            "connected": [
                {"note": self._display(n), "exists": n in self.nodes}
                for n in connected[:limit]
            ],
            "edges": [[self._display(a), self._display(b)] for a, b in edges[: limit * 3]],
        }

    def search(self, query: str, limit: int = 8) -> dict:
        q = query.strip().lower()
        name_hits: list[dict] = []
        if q:
            for key, refs in self.nodes.items():
                if q in key:
                    for r in refs:
                        name_hits.append({
                            "note": r.name, "vault": r.vault,
                            "match": "name",
                            "links": len(self.outgoing.get(key, [])),
                            "backlinks": len(self.backlinks.get(key, [])),
                        })
        # Rank name hits by connectedness (hub notes first) — the big dots.
        name_hits.sort(key=lambda h: h["links"] + h["backlinks"], reverse=True)
        content_hits, engine = _content_search(query, [Path(r) for r in self.roots], limit)
        return {
            "query": query,
            "name_matches": name_hits[:limit],
            "content_matches": content_hits[:limit],
            "content_engine": engine,
        }

    def stats(self) -> dict:
        by_vault: dict[str, int] = defaultdict(int)
        for refs in self.nodes.values():
            for r in refs:
                by_vault[r.vault] += 1
        total_edges = sum(len(v) for v in self.outgoing.values())
        dangling = sum(1 for k in self.backlinks if k not in self.nodes)
        hubs = sorted(
            self.nodes.keys(),
            key=lambda k: len(self.outgoing.get(k, [])) + len(self.backlinks.get(k, [])),
            reverse=True,
        )[:10]
        return {
            "notes": sum(len(v) for v in self.nodes.values()),
            "notes_by_vault": dict(by_vault),
            "edges": total_edges,
            "dangling_targets": dangling,
            "top_hubs": [self._display(h) for h in hubs],
            "roots": self.roots,
        }

    def _display(self, key: str) -> str:
        refs = self.nodes.get(key)
        return refs[0].name if refs else key


# ── normalisation + parsing ─────────────────────────────────────────────────


def _norm(name: str) -> str:
    """Obsidian resolves `[[Name]]` by filename stem, case-insensitively. Strip
    alias (`|`), heading/block (`#`), any path, the .md extension → lowercase."""
    n = name.split("|", 1)[0].split("#", 1)[0].strip()
    n = n.replace("\\", "/").rsplit("/", 1)[-1]
    if n.lower().endswith(".md"):
        n = n[:-3]
    return n.strip().lower()


def _vault_paths(settings: Settings) -> list[Path]:
    raw = (settings.vault_dirs or "").strip()
    out: list[Path] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser()
        if p.is_dir():
            out.append(p.resolve())
        else:
            log.warning("vault.dir_missing", path=str(p))
    return out


def _iter_md(vaults: list[Path]):
    for v in vaults:
        for p in sorted(v.rglob("*.md")):
            if p.is_file():
                yield v.name, p


def _signature(vaults: list[Path]) -> list:
    """Cheap freshness key: (file_count, total_bytes, max_mtime). Stat-only —
    no content read — so it's fast even across thousands of notes."""
    fc = 0
    total = 0
    newest = 0.0
    for _vname, p in _iter_md(vaults):
        try:
            st = p.stat()
        except OSError:
            continue
        fc += 1
        total += st.st_size
        if st.st_mtime > newest:
            newest = st.st_mtime
    return [fc, total, round(newest, 3)]


def build_graph(vaults: list[Path]) -> VaultGraph:
    nodes: dict[str, list[NoteRef]] = {}
    outgoing: dict[str, list[str]] = {}
    backlinks: dict[str, set[str]] = defaultdict(set)
    fc = 0
    total = 0
    newest = 0.0
    for vault_name, path in _iter_md(vaults):
        stem = path.stem
        key = _norm(stem)
        nodes.setdefault(key, []).append(NoteRef(name=stem, vault=vault_name, path=str(path)))
        try:
            st = path.stat()
            fc += 1
            total += st.st_size
            newest = max(newest, st.st_mtime)
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            outgoing.setdefault(key, [])
            continue
        targets: list[str] = []
        seen: set[str] = set()
        for m in _WIKILINK.finditer(text):
            tgt = _norm(m.group(1))
            if tgt and tgt != key and tgt not in seen:
                seen.add(tgt)
                targets.append(tgt)
                backlinks[tgt].add(key)
        outgoing[key] = targets
    graph = VaultGraph(
        nodes=nodes,
        outgoing=outgoing,
        backlinks={k: sorted(v) for k, v in backlinks.items()},
        roots=[str(v) for v in vaults],
        signature=[fc, total, round(newest, 3)],
    )
    log.info("vault.graph_built", notes=sum(len(v) for v in nodes.values()),
             edges=sum(len(v) for v in outgoing.values()), vaults=len(vaults))
    return graph


# ── content search (ripgrep-preferred, name-only fallback) ──────────────────


def _content_search(query: str, roots: list[Path], limit: int) -> tuple[list[dict], str]:
    """Grep note *bodies* for the query. Uses ripgrep (--json → robust across
    Windows drive-letter paths) when present; returns (hits, engine). Absent rg
    → ([], "none") and the caller still has instant name matches from the graph."""
    if not query.strip() or not roots:
        return [], "none"
    rg = shutil.which("rg")
    if not rg:
        return [], "unavailable (install ripgrep for content search)"
    cmd = [rg, "--json", "-i", "--max-count", "2", "-g", "*.md", "-e", query]
    cmd += [str(r) for r in roots]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (subprocess.SubprocessError, OSError) as e:
        log.info("vault.rg_failed", error=str(e))
        return [], "error"
    hits: list[dict] = []
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj.get("data", {})
        path = (data.get("path") or {}).get("text", "")
        snippet = (data.get("lines") or {}).get("text", "").strip()
        hits.append({
            "note": Path(path).stem,
            "line": data.get("line_number"),
            "snippet": snippet[:200],
            "match": "content",
        })
        if len(hits) >= limit * 3:
            break
    return hits, "ripgrep"


# ── cache + singleton access ─────────────────────────────────────────────────


def _cache_path(settings: Settings) -> Path:
    cache = settings.cache_dir.expanduser()
    if not cache.is_absolute():
        cache = (Path.cwd() / cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    return cache / _CACHE_FILE


def _load_cache(path: Path, expect_sig: list) -> VaultGraph | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if raw.get("signature") != expect_sig:
        return None
    nodes = {
        k: [NoteRef(**r) for r in refs] for k, refs in raw.get("nodes", {}).items()
    }
    return VaultGraph(
        nodes=nodes,
        outgoing=raw.get("outgoing", {}),
        backlinks=raw.get("backlinks", {}),
        roots=raw.get("roots", []),
        signature=raw.get("signature", []),
    )


def _save_cache(path: Path, graph: VaultGraph) -> None:
    payload = {
        "nodes": {k: [asdict(r) for r in refs] for k, refs in graph.nodes.items()},
        "outgoing": graph.outgoing,
        "backlinks": graph.backlinks,
        "roots": graph.roots,
        "signature": graph.signature,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def get_graph(settings: Settings | None = None, *, refresh: bool = False) -> VaultGraph:
    """Return the vault graph, building/caching as needed.

    In-memory singleton for the process. On first use (or refresh) it checks the
    cheap stat-signature: unchanged → load the JSON cache; changed → rebuild
    (reads note bodies for links) and re-cache. Raises if no vaults configured.
    """
    global _cached
    settings = settings or get_settings()
    vaults = _vault_paths(settings)
    if not vaults:
        raise RuntimeError(
            "no vaults configured — set LUMOS_VAULT_DIRS to comma-separated "
            "Obsidian folder paths, then reindex."
        )
    sig = _signature(vaults)
    if not refresh and _cached is not None and _cached.signature == sig:
        return _cached
    cache_path = _cache_path(settings)
    if not refresh:
        cached = _load_cache(cache_path, sig)
        if cached is not None:
            _cached = cached
            return cached
    graph = build_graph(vaults)
    _save_cache(cache_path, graph)
    _cached = graph
    return graph


def reindex(settings: Settings | None = None) -> dict:
    """Force a full rebuild (after big Obsidian edits). Returns stats."""
    graph = get_graph(settings, refresh=True)
    return graph.stats()
