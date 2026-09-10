from typing import Sequence

import pytest

from northstar.embeddings import NeuralEmbeddingProvider
from northstar.ingestion import Chunk
from northstar.retrieval import SemanticSearchIndex, cosine_similarity


class FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(
        self,
        texts: Sequence[str],
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ):
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]


def make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        metadata={
            "document_title": "Neural Test Policy",
            "source_filename": "neural-test.md",
            "section_title": "Testing",
            "chunk_index": index,
        },
    )


def test_neural_embedding_dimensions_and_stable_shape() -> None:
    provider = NeuralEmbeddingProvider(model_name="test-model", model=FakeSentenceTransformer())

    first = provider.embed_text("same text")
    second = provider.embed_text("same text")

    assert provider.model_name == "test-model"
    assert provider.dimension == 4
    assert len(first) == 4
    assert first == second


def test_neural_batch_embedding() -> None:
    provider = NeuralEmbeddingProvider(model=FakeSentenceTransformer())

    embeddings = provider.embed_texts(["one", "two"])

    assert len(embeddings) == 2
    assert all(len(embedding) == 4 for embedding in embeddings)


def test_neural_provider_works_with_semantic_search_index() -> None:
    provider = NeuralEmbeddingProvider(model=FakeSentenceTransformer())
    index = SemanticSearchIndex(
        [make_chunk(0, "short"), make_chunk(1, "a much longer text")],
        provider,
    )

    results = index.search("a much longer text", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_index == 1
    assert results[0].document_title == "Neural Test Policy"
    assert results[0].source_filename == "neural-test.md"


def test_neural_search_preserves_cosine_and_top_k() -> None:
    provider = NeuralEmbeddingProvider(model=FakeSentenceTransformer())
    index = SemanticSearchIndex(
        [make_chunk(0, "one"), make_chunk(1, "longer")],
        provider,
    )

    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert len(index.search("query", top_k=1)) == 1
