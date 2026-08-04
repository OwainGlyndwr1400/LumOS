"""Identity corpus loader — dream-codex .md → IdentityChunk (2026-07-15).

Offline: writes tiny .md/.txt/.csv files to a tmp dir and checks the loader
produces identity-shaped chunks, chunks long prose by paragraph, tags dreams
correctly, and skips CSV (that's knowledge, not identity).
"""

from lumos_node.knowledge.corpus import iter_identity_corpus_chunks
from lumos_node.memory.identity import IdentityChunk


def _write(d, name, text):
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def test_md_becomes_identity_chunks(tmp_path):
    _write(tmp_path, "dream_01.md", "First dream paragraph about the Lion gate.\n\nSecond stanza, Regulus rising.")
    chunks = list(iter_identity_corpus_chunks(tmp_path))
    assert chunks and all(isinstance(c, IdentityChunk) for c in chunks)
    c = chunks[0]
    assert c.conversation_id == "dreamcodex:dream_01"
    assert c.roles == ["dream"]
    assert c.node_ids == [c.chunk_id]
    assert "Lion gate" in c.text


def test_long_prose_splits_into_multiple_chunks(tmp_path):
    # 6 paragraphs of ~800 chars each → must exceed the 2000-char target > once.
    paras = "\n\n".join(f"Paragraph {i} " + ("dreamword " * 100) for i in range(6))
    _write(tmp_path, "codex.md", paras)
    chunks = list(iter_identity_corpus_chunks(tmp_path))
    assert len(chunks) >= 2
    # Every chunk stays a reasonable size (target 2000 + one trailing paragraph).
    assert all(len(c.text) < 4000 for c in chunks)
    # Section subjects are distinct per chunk.
    assert len({c.conversation_title for c in chunks}) == len(chunks)


def test_csv_is_skipped_belongs_to_knowledge(tmp_path):
    _write(tmp_path, "theorems.csv", "name,equation\nLion,L=sqrt(3)/(2phi)\n")
    _write(tmp_path, "dream.md", "A dream about the grid.")
    ids = {c.conversation_id for c in iter_identity_corpus_chunks(tmp_path)}
    assert ids == {"dreamcodex:dream"}  # csv contributed nothing


def test_stable_chunk_ids_are_deterministic(tmp_path):
    _write(tmp_path, "d.md", "Stable dream content for hashing.")
    a = list(iter_identity_corpus_chunks(tmp_path))
    b = list(iter_identity_corpus_chunks(tmp_path))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_empty_dir_yields_nothing(tmp_path):
    assert list(iter_identity_corpus_chunks(tmp_path)) == []
