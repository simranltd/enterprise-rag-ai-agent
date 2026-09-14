from pathlib import Path

from northstar.ingestion import Chunk
from northstar.persistence import ChromaVectorStore


def test_chroma_persists_and_reingestion_is_idempotent(tmp_path: Path) -> None:
    directory = tmp_path / "chroma"
    chunk = Chunk(
        "persistent content",
        {
            "document_title": "Test",
            "source_filename": "test.md",
            "document_version": "1",
            "section_title": "Section",
            "chunk_index": 0,
            "embedding_model": "test",
            "embedding_dimension": 2,
        },
    )
    first = ChromaVectorStore(directory)
    first.upsert_chunks([chunk], [[1.0, 0.0]])
    assert first.count() == 1

    reopened = ChromaVectorStore(directory)
    reopened.upsert_chunks([chunk], [[1.0, 0.0]])
    assert reopened.count() == 1
    result = reopened.search([1.0, 0.0], top_k=1)[0]
    assert result["chunk_text"] == "persistent content"


def test_sync_removes_stale_chunks_and_preserves_current_ids(tmp_path: Path) -> None:
    directory = tmp_path / "chroma"
    store = ChromaVectorStore(directory)
    first = Chunk(
        "keep this",
        {
            "document_title": "Test",
            "source_filename": "test.md",
            "document_version": "1",
            "section_title": "Current",
            "chunk_index": 0,
        },
    )
    stale = Chunk(
        "remove this",
        {
            "document_title": "Test",
            "source_filename": "test.md",
            "document_version": "1",
            "section_title": "Removed",
            "chunk_index": 1,
        },
    )
    store.sync_chunks([first, stale], [[1.0, 0.0], [0.0, 1.0]])
    first_id = store.collection.get()["ids"][0]

    store.sync_chunks([first], [[1.0, 0.0]])

    assert store.count() == 1
    assert store.collection.get()["ids"] == [first_id]
    assert store.search([1.0, 0.0], top_k=2)[0]["chunk_text"] == "keep this"


def test_sync_reindexes_changed_chunk_without_duplicate_records(tmp_path: Path) -> None:
    directory = tmp_path / "chroma"
    store = ChromaVectorStore(directory)
    original = Chunk(
        "old policy text",
        {
            "source_filename": "policy.md",
            "document_version": "1",
            "chunk_index": 0,
        },
    )
    updated = Chunk(
        "new policy text",
        {
            "source_filename": "policy.md",
            "document_version": "1",
            "chunk_index": 0,
        },
    )

    store.sync_chunks([original], [[1.0, 0.0]])
    store.sync_chunks([updated], [[1.0, 0.0]])
    store.sync_chunks([updated], [[1.0, 0.0]])

    assert store.count() == 1
    assert store.search([1.0, 0.0], top_k=1)[0]["chunk_text"] == "new policy text"
