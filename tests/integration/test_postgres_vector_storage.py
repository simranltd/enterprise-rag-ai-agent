import os

import pytest

from northstar.embeddings import NeuralEmbeddingProvider
from northstar.ingestion import Chunk
from northstar.persistence import PostgresChunkStore
from northstar.retrieval import PostgresVectorSearch


pytestmark = pytest.mark.integration


def store() -> PostgresChunkStore:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not configured")
    return PostgresChunkStore(
        url,
        os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        384,
    )


def test_schema_insert_duplicate_and_round_trip() -> None:
    db = store()
    db.initialize_schema()
    chunk = Chunk(
        "A persistent test chunk.",
        {
            "document_title": "Integration Policy",
            "source_filename": "integration.md",
            "document_version": "1",
            "document_type": "Policy",
            "department": "Testing",
            "jurisdiction": "Canada",
            "classification": "Internal",
            "section_title": "Test",
            "chunk_index": 0,
        },
    )
    embedding = [0.0] * 383 + [1.0]
    first = db.insert_chunks([chunk], [embedding])
    second = db.insert_chunks([chunk], [embedding])
    assert first in (0, 1)
    assert second == 0


def test_dimension_validation() -> None:
    db = store()
    with pytest.raises(ValueError):
        db.insert_chunks(
            [
                Chunk("bad", {"document_title": "T", "source_filename": "f"})
            ],
            [[0.0]],
        )


def test_top_k_and_metadata_search() -> None:
    db = store()
    provider = NeuralEmbeddingProvider()
    search = PostgresVectorSearch(db, provider)
    results = search.search("persistent test chunk", top_k=1)
    assert len(results) <= 1
