"""Compare one-query-embedding in-memory and persistent Chroma retrieval."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import NeuralEmbeddingProvider  # noqa: E402
from northstar.ingestion import chunk_document, load_markdown_document  # noqa: E402
from northstar.persistence import ChromaVectorStore  # noqa: E402
from northstar.retrieval import ChromaVectorSearch, SemanticSearchIndex, cosine_similarity  # noqa: E402

QUERIES = {
    "A": "Can I work from another country?",
    "B": "Am I allowed to work overseas?",
    "C": "What approval is required for an expense around $5,000?",
    "D": "Who authorizes a large purchase?",
    "E": "Is MFA required?",
    "F": "Do employees need a second authentication factor?",
    "G": "What checks are required for a high-risk client?",
    "H": "What extra checks apply to a risky customer?",
}


def chunks_from_documents() -> list:
    chunks = []
    for path in sorted((ROOT / "data" / "sample_documents").glob("*.md")):
        chunks.extend(chunk_document(load_markdown_document(path)))
    return chunks


def memory_search(index: SemanticSearchIndex, query_embedding: list, top_k: int = 3) -> list:
    ranked = sorted(
        (
            (cosine_similarity(query_embedding, embedding), index_number, chunk)
            for index_number, (embedding, chunk) in enumerate(
                zip(index.embeddings, index.chunks)
            )
        ),
        key=lambda value: (-value[0], value[1]),
    )[:top_k]
    return [
        {
            "rank": rank,
            "score": score,
            "document_title": chunk.metadata["document_title"],
            "section_title": chunk.metadata["section_title"],
            "chunk_index": chunk.metadata["chunk_index"],
        }
        for rank, (score, _, chunk) in enumerate(ranked, start=1)
    ]


def print_results(label: str, results: list, score_name: str) -> None:
    print(f"  {label}")
    for result in results:
        score = result.get("score", result.get("cosine_similarity"))
        print(
            f"    {result['rank']}. {result['document_title']} / "
            f"{result['section_title']} / chunk {result['chunk_index']} "
            f"({score_name}={score:.4f})"
        )


def main() -> None:
    provider = NeuralEmbeddingProvider()
    chunks = chunks_from_documents()
    for chunk in chunks:
        chunk.metadata["embedding_model"] = provider.model_name
        chunk.metadata["embedding_dimension"] = provider.dimension
    index = SemanticSearchIndex(chunks, provider)
    chroma = ChromaVectorSearch(ChromaVectorStore())

    for label, query in QUERIES.items():
        query_embedding = provider.embed_text(query)
        memory_results = memory_search(index, query_embedding)
        chroma_results = chroma.search(query_embedding)
        chroma_view = [
            {
                "rank": result.rank,
                "cosine_similarity": result.cosine_similarity,
                "document_title": result.document_title,
                "section_title": result.section_title,
                "chunk_index": result.chunk_index,
            }
            for result in chroma_results
        ]
        memory_keys = [
            (r["document_title"], r["section_title"], r["chunk_index"])
            for r in memory_results
        ]
        chroma_keys = [
            (r["document_title"], r["section_title"], r["chunk_index"])
            for r in chroma_view
        ]
        print(f"\n{label}. {query}")
        print_results("IN-MEMORY NEURAL", memory_results, "similarity")
        print_results("CHROMA COSINE", chroma_view, "similarity")
        print(f"  Ranking: {'equivalent' if memory_keys == chroma_keys else 'different'}")


if __name__ == "__main__":
    main()
