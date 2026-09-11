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
