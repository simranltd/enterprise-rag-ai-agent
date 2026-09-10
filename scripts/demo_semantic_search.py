"""Run transparent, in-memory semantic searches over Northstar documents."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import LocalHashEmbeddingProvider  # noqa: E402
from northstar.ingestion import chunk_document, load_markdown_document  # noqa: E402
from northstar.retrieval import SemanticSearchIndex  # noqa: E402


QUERIES = [
    "Can I work from another country?",
    "What approvals are needed for expenses over $5,000?",
    "What happens if an employee loses MFA access?",
    "What is required for a high-risk client?",
]


def build_index() -> SemanticSearchIndex:
    chunks = []
    for path in sorted((ROOT / "data" / "sample_documents").glob("*.md")):
        chunks.extend(chunk_document(load_markdown_document(path)))
    return SemanticSearchIndex(chunks, LocalHashEmbeddingProvider())


def main() -> None:
    index = build_index()
    print(f"Indexed {len(index.chunks)} chunks in memory.")
    for query in QUERIES:
        print(f"\nQuery: {query}")
        for result in index.search(query, top_k=3):
            print(
                f"  {result.rank}. score={result.similarity_score:.4f} "
                f"{result.source_filename} / {result.section_title} "
                f"(chunk {result.chunk_index})"
            )
            print(f"     {result.chunk_text_preview.replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
