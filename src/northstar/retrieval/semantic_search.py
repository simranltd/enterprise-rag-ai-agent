"""Explicit in-memory semantic search using cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence

from northstar.embeddings import EmbeddingProvider
from northstar.ingestion import Chunk


@dataclass(frozen=True)
class SearchResult:
    """A ranked match returned by the in-memory semantic index."""

    rank: int
    similarity_score: float
    document_title: str
    source_filename: str
    section_title: str
    chunk_index: int
    chunk_text_preview: str
    chunk: Chunk


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine of the angle between two equal-length vectors."""

    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right))
    return dot_product / (left_norm * right_norm)


class SemanticSearchIndex:
    """An in-memory chunk index that owns embeddings and ranked search."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        embedding_provider: EmbeddingProvider,
        preview_length: int = 150,
    ) -> None:
        if preview_length <= 0:
            raise ValueError("preview_length must be greater than zero")
        self.embedding_provider = embedding_provider
        self.preview_length = preview_length
        self.chunks = list(chunks)
        self.embeddings = embedding_provider.embed_texts(
            [
                f"{chunk.metadata.get('section_title', '')}\n{chunk.text}"
                for chunk in self.chunks
            ]
        )

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            return []

        query_embedding = self.embedding_provider.embed_text(query)
        ranked = sorted(
            (
                (cosine_similarity(query_embedding, embedding), index, chunk)
                for index, (embedding, chunk) in enumerate(
                    zip(self.embeddings, self.chunks)
                )
            ),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]
        return [
            SearchResult(
                rank=rank,
                similarity_score=score,
                document_title=str(chunk.metadata.get("document_title", "")),
                source_filename=str(chunk.metadata.get("source_filename", "")),
                section_title=str(chunk.metadata.get("section_title", "")),
                chunk_index=int(chunk.metadata.get("chunk_index", index)),
                chunk_text_preview=chunk.text[: self.preview_length],
                chunk=chunk,
            )
            for rank, (score, index, chunk) in enumerate(ranked, start=1)
        ]
