from northstar.embeddings import LocalHashEmbeddingProvider
from northstar.ingestion import Chunk
from northstar.retrieval import SemanticSearchIndex, cosine_similarity
import pytest


def make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        metadata={
            "document_title": "Test Policy",
            "source_filename": "test-policy.md",
            "section_title": f"Section {index}",
            "chunk_index": index,
        },
    )


def test_embedding_result_shape() -> None:
    provider = LocalHashEmbeddingProvider(dimension=32)

    embedding = provider.embed_text("Northstar policy")
    batch = provider.embed_texts(["one", "two"])

    assert len(embedding) == 32
    assert len(batch) == 2
    assert all(len(vector) == 32 for vector in batch)


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_result_ordering_and_top_k() -> None:
    provider = LocalHashEmbeddingProvider()
    chunks = [
        make_chunk(0, "expense approvals and finance"),
        make_chunk(1, "remote work outside Canada"),
        make_chunk(2, "expense approval threshold"),
    ]
    results = SemanticSearchIndex(chunks, provider).search(
        "What approvals are needed for expenses?", top_k=2
    )

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]
    assert results[0].chunk_index in (0, 2)
    assert results[0].similarity_score >= results[1].similarity_score


def test_metadata_preservation_and_preview() -> None:
    chunk = make_chunk(4, "A" * 200)
    result = SemanticSearchIndex([chunk], LocalHashEmbeddingProvider()).search("A")[0]

    assert result.document_title == "Test Policy"
    assert result.source_filename == "test-policy.md"
    assert result.section_title == "Section 4"
    assert result.chunk_index == 4
    assert len(result.chunk_text_preview) == 150


def test_empty_query_handling() -> None:
    index = SemanticSearchIndex(
        [make_chunk(0, "content")], LocalHashEmbeddingProvider()
    )

    with pytest.raises(ValueError, match="query cannot be empty"):
        index.search("   ")
