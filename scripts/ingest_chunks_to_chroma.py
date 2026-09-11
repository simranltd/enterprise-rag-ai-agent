"""Ingest Northstar chunks and neural embeddings into persistent Chroma."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import DEFAULT_MODEL_NAME, NeuralEmbeddingProvider  # noqa: E402
from northstar.ingestion import chunk_document, load_markdown_document  # noqa: E402
from northstar.persistence import ChromaVectorStore  # noqa: E402


def main() -> None:
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    provider = NeuralEmbeddingProvider(model_name=model_name)
    store = ChromaVectorStore(
        os.environ.get("CHROMA_PERSISTENCE_DIRECTORY", "data/chroma"),
        os.environ.get("CHROMA_COLLECTION_NAME", "northstar_document_chunks"),
    )
    chunks = []
    for path in sorted((ROOT / "data" / "sample_documents").glob("*.md")):
        chunks.extend(chunk_document(load_markdown_document(path)))
    embeddings = provider.embed_texts(
        [f"{chunk.metadata.get('section_title', '')}\n{chunk.text}" for chunk in chunks]
    )
    for chunk in chunks:
        chunk.metadata["embedding_model"] = model_name
        chunk.metadata["embedding_dimension"] = provider.dimension
    before = store.count()
    upserted = store.upsert_chunks(chunks, embeddings)
    print(f"Documents processed: {len(list((ROOT / 'data' / 'sample_documents').glob('*.md')))}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Chunks newly inserted/upserted: {upserted}")
    print(f"Collection count before: {before}")
    print(f"Collection count after: {store.count()}")


if __name__ == "__main__":
    main()
