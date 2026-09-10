"""Compare hash and neural semantic retrieval over the same chunks."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from northstar.embeddings import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    LocalHashEmbeddingProvider,
    NeuralEmbeddingProvider,
)
from northstar.ingestion import chunk_document, load_markdown_document  # noqa: E402
from northstar.retrieval import SemanticSearchIndex  # noqa: E402


QUERIES = {
    "A": "Can I work from another country?",
    "B": "Am I allowed to perform my job while travelling overseas?",
    "C": "What approvals are needed for expenses over $5,000?",
    "D": "Who needs to authorize a large business purchase?",
    "E": "What happens if an employee loses MFA access?",
    "F": "I cannot use my second authentication factor. What should I do?",
    "G": "What is required for a high-risk client?",
    "H": "What extra checks are necessary for a customer considered risky?",
}


def load_chunks() -> list:
    chunks = []
    for path in sorted((ROOT / "data" / "sample_documents").glob("*.md")):
        chunks.extend(chunk_document(load_markdown_document(path)))
    return chunks


def timed_search(index: SemanticSearchIndex, query: str, top_k: int = 3):
    query_start = perf_counter()
    results = index.search(query, top_k=top_k)
    query_time = perf_counter() - query_start
    return results, query_time


def print_results(results) -> None:
    for result in results:
        print(
            f"  {result.rank}. {result.source_filename} / "
            f"{result.section_title} / chunk {result.chunk_index} "
            f"(score={result.similarity_score:.4f})"
        )


def main() -> None:
    chunks = load_chunks()
    hash_provider = LocalHashEmbeddingProvider()

    neural_load_start = perf_counter()
    neural_provider = NeuralEmbeddingProvider(model_name=DEFAULT_MODEL_NAME)
    _ = neural_provider.model
    neural_load_time = perf_counter() - neural_load_start

    hash_embedding_start = perf_counter()
    hash_index = SemanticSearchIndex(chunks, hash_provider)
    hash_embedding_time = perf_counter() - hash_embedding_start

    neural_embedding_start = perf_counter()
    neural_index = SemanticSearchIndex(chunks, neural_provider)
    neural_embedding_time = perf_counter() - neural_embedding_start

    print(f"Neural model: {neural_provider.model_name}")
    print(f"Neural dimensions: {neural_provider.dimension}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Model loading: {neural_load_time:.3f}s")
    print(f"Hash document embedding/indexing: {hash_embedding_time:.3f}s")
    print(f"Neural document embedding/indexing: {neural_embedding_time:.3f}s")

    for label, query in QUERIES.items():
        hash_results, hash_query_time = timed_search(hash_index, query)
        neural_results, neural_query_time = timed_search(neural_index, query)
        print(f"\n{label}. {query}")
        print(f"  HASH BASELINE (query/search: {hash_query_time:.6f}s)")
        print_results(hash_results)
        print(f"  NEURAL EMBEDDINGS (query/search: {neural_query_time:.6f}s)")
        print_results(neural_results)


if __name__ == "__main__":
    main()
