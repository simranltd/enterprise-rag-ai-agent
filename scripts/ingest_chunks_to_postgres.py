"""Ingest Northstar chunks and neural embeddings into PostgreSQL."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import DEFAULT_MODEL_NAME, NeuralEmbeddingProvider  # noqa: E402
from northstar.ingestion import chunk_document, load_markdown_document  # noqa: E402
from northstar.persistence import PostgresChunkStore  # noqa: E402


def main() -> None:
    connection_string = os.environ.get("DATABASE_URL", "")
    if not connection_string:
        raise RuntimeError("DATABASE_URL must be set")
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    provider = NeuralEmbeddingProvider(model_name=model_name)
    store = PostgresChunkStore(connection_string, model_name, provider.dimension)
    store.initialize_schema()

    chunks = []
    for path in sorted((ROOT / "data" / "sample_documents").glob("*.md")):
        chunks.extend(chunk_document(load_markdown_document(path)))
    embeddings = provider.embed_texts(
        [f"{chunk.metadata.get('section_title', '')}\n{chunk.text}" for chunk in chunks]
    )
    started = perf_counter()
    inserted = store.insert_chunks(chunks, embeddings)
    print(f"Processed chunks: {len(chunks)}")
    print(f"Inserted rows: {inserted}")
    print(f"Rows currently stored: {store.count()}")
    print(f"Database insertion time: {perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
