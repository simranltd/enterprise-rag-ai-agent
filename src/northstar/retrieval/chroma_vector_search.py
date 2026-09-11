"""Retrieval adapter for persistent Chroma cosine-distance search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from northstar.persistence.chroma_store import ChromaVectorStore


@dataclass(frozen=True)
class ChromaSearchResult:
    rank: int
    distance: float
    cosine_similarity: float
    document_title: str
    source_filename: str
    section_title: str
    chunk_index: int
    chunk_text: str
    chunk_text_preview: str
    metadata: dict


class ChromaVectorSearch:
    """Search Chroma with caller-supplied query embeddings.

    Chroma returns cosine distance because the collection is configured with
    cosine space. For normalized vectors, cosine similarity is 1 - distance.
    The distance is retained so it is not confused with probability.
    """

    def __init__(
        self,
        store: ChromaVectorStore,
        preview_length: int = 150,
    ) -> None:
        if preview_length <= 0:
            raise ValueError("preview_length must be greater than zero")
        self.store = store
        self.preview_length = preview_length

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[ChromaSearchResult]:
        return [
            ChromaSearchResult(
                rank=row["rank"],
                distance=row["distance"],
                cosine_similarity=1.0 - row["distance"],
                document_title=row["metadata"]["document_title"],
                source_filename=row["metadata"]["source_filename"],
                section_title=row["metadata"]["section_title"],
                chunk_index=row["metadata"]["chunk_index"],
                chunk_text=row["chunk_text"],
                chunk_text_preview=row["chunk_text"][: self.preview_length],
                metadata=row["metadata"],
            )
            for row in self.store.search(query_embedding, top_k)
        ]
