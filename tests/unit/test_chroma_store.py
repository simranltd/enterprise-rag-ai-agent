from pathlib import Path

from northstar.ingestion import Chunk
from northstar.persistence.chroma_store import (
    ChromaVectorStore,
    content_hash,
    deterministic_chunk_id,
)


def test_content_hash_and_id_are_deterministic() -> None:
    chunk = Chunk(
        "same chunk",
        {
            "source_filename": "policy.md",
            "document_version": "1.0",
            "chunk_index": 2,
        },
    )
    assert content_hash(chunk.text) == content_hash(chunk.text)
    assert deterministic_chunk_id(chunk) == deterministic_chunk_id(chunk)


def test_id_changes_when_identity_changes() -> None:
    base = Chunk("same", {"source_filename": "a.md", "document_version": "1", "chunk_index": 0})
    changed = Chunk("same", {"source_filename": "a.md", "document_version": "2", "chunk_index": 0})
    assert deterministic_chunk_id(base) != deterministic_chunk_id(changed)


def test_metadata_handling(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma")
    chunk = Chunk(
        "text",
        {
            "document_title": "Policy",
            "source_filename": "policy.md",
            "document_version": "1",
            "document_type": "Policy",
            "department": "Finance",
            "jurisdiction": "Canada",
            "classification": "Internal",
            "section_title": "Approvals",
            "chunk_index": 0,
            "embedding_model": "test",
            "embedding_dimension": 384,
        },
    )
    store.upsert_chunks([chunk], [[1.0, 0.0]])
    metadata = store.collection.get(include=["metadatas"])["metadatas"][0]
    assert metadata["document_title"] == "Policy"
    assert metadata["section_title"] == "Approvals"
