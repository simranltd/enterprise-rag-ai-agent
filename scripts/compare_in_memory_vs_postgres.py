"""Compare exact in-memory and PostgreSQL pgvector neural retrieval."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import DEFAULT_MODEL_NAME, NeuralEmbeddingProvider  # noqa: E402
from northstar.ingestion import chunk_document, load_markdown_document  # noqa: E402
from northstar.persistence import PostgresChunkStore  # noqa: E402
from northstar.retrieval import PostgresVectorSearch, SemanticSearchIndex  # noqa: E402

QUERIES = [
    "Can I work from another country?",
    "What approvals are needed for expenses over $5,000?",
    "What happens if an employee loses MFA access?",
    "What is required for a high-risk client?",
]


def main() -> None:
    connection_string = os.environ.get("DATABASE_URL", "")
    if not connection_string:
        raise RuntimeError("DATABASE_URL must be set")
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    provider = NeuralEmbeddingProvider(model_name=model_name)
    chunks = []
    for path in sorted((ROOT / "data" / "sample_documents").glob("*.md")):
        chunks.extend(chunk_document(load_markdown_document(path)))
    memory = SemanticSearchIndex(chunks, provider)
    postgres = PostgresVectorSearch(
        PostgresChunkStore(connection_string, model_name, provider.dimension),
        provider,
    )
    for query in QUERIES:
        memory_results = memory.search(query)
        postgres_results = postgres.search(query)
        memory_keys = [
            (result.source_filename, result.section_title, result.chunk_index)
            for result in memory_results
        ]
        postgres_keys = [
            (result.source_filename, result.section_title, result.chunk_index)
            for result in postgres_results
        ]
        status = "equivalent" if memory_keys == postgres_keys else "materially different"
        print(f"\nQuery: {query}\nRanking comparison: {status}")
        print("  In-memory:", memory_keys)
        print("  PostgreSQL:", postgres_keys)


if __name__ == "__main__":
    main()
