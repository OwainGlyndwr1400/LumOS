"""Obsidian vault-graph tools — Lumos's associative third memory.

These traverse the operator's `[[wikilink]]` graph across the configured vaults
(LUMOS_VAULT_DIRS) WITHOUT any embedding — a live, always-current brain-map that
complements the semantic FAISS lanes. Read-only: search, read, follow links.

Typical flow the model follows: vault_search to find an entry note → vault_read
to see its content + what it links to → vault_links to walk deeper into the
connected memories.
"""

from __future__ import annotations

from typing import Any

from ..knowledge import vault_graph
from ..log import get_logger
from . import register

log = get_logger(__name__)


def _graph_or_error() -> Any:
    """Return the graph, or a dict the tool can hand straight back on failure."""
    try:
        return vault_graph.get_graph()
    except RuntimeError as e:
        return {"error": str(e)}


@register(
    name="vault_search",
    description=(
        "Search Lumos's Obsidian vault-graph (the operator's linked notes — dream "
        "pings, research, the RHC framework) for a topic. Returns matching notes by "
        "name (hub notes first) and by content. This is the ENTRY POINT: use it to "
        "find which note to open, then vault_read it and vault_links to go deeper. "
        "Separate from search_memory/search_knowledge (those are the FAISS lanes)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Topic / keyword / note name to look for."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Max hits per kind (default 8)."},
        },
        "required": ["query"],
    },
)
def vault_search(query: str, limit: int = 8) -> dict:
    graph = _graph_or_error()
    if isinstance(graph, dict):
        return graph
    return graph.search(query, limit=limit)


@register(
    name="vault_read",
    description=(
        "Read one note from the vault-graph by name: its text PLUS its outgoing "
        "[[links]] and its backlinks (which notes point at it). This is how Lumos "
        "'clicks a dot' — see the memory and every memory it's wired to. Follow the "
        "listed links with another vault_read, or vault_links to expand the web."
    ),
    parameters={
        "type": "object",
        "properties": {
            "note": {"type": "string", "description": "Note name (filename stem, e.g. 'Abhimanyu')."},
            "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000, "description": "Cap note text (default 8000)."},
        },
        "required": ["note"],
    },
)
def vault_read(note: str, max_chars: int = 8000) -> dict:
    graph = _graph_or_error()
    if isinstance(graph, dict):
        return graph
    result = graph.read_note(note, max_chars=max_chars)
    if result is None:
        return {"error": f"no note named {note!r} in the vault-graph", "note": note}
    return result


@register(
    name="vault_links",
    description=(
        "Walk the vault-graph outward from a note to see its connected web of "
        "memories (neighbours up to `depth` hops, following both [[links]] and "
        "backlinks). Use to 'go deeper' — trace how a concept threads through the "
        "operator's notes, or map a note's neighbourhood before reading the pieces."
    ),
    parameters={
        "type": "object",
        "properties": {
            "note": {"type": "string", "description": "Note name to start from."},
            "depth": {"type": "integer", "minimum": 1, "maximum": 3, "description": "Hops to expand (default 1)."},
        },
        "required": ["note"],
    },
)
def vault_links(note: str, depth: int = 1) -> dict:
    graph = _graph_or_error()
    if isinstance(graph, dict):
        return graph
    return graph.neighbors(note, depth=depth)


@register(
    name="vault_reindex",
    description=(
        "Rebuild the vault-graph from disk — run after the operator has added or "
        "heavily edited notes in Obsidian so new links/notes are picked up. Returns "
        "graph stats (note/edge counts, top hub notes). Cheap: parses links, no embedding."
    ),
    parameters={"type": "object", "properties": {}},
)
def vault_reindex() -> dict:
    try:
        return vault_graph.reindex()
    except RuntimeError as e:
        return {"error": str(e)}
