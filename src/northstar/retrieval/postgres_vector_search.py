"""PostgreSQL-backed neural vector search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from northstar.embeddings import EmbeddingProvider
from northstar.persistence import PostgresChunkStore


@dataclass(frozen=True)
class PostgresSearchResult:
    rank: int
    similarity_score: float
    document_title: str
    source_filename: str
    section_title: str
    chunk_index: int
    chunk_text_preview: str


class PostgresVectorSearch:
    def __init__(
        self,
        store: PostgresChunkStore,
        embedding_provider: EmbeddingProvider,
        preview_length: int = 150,
    ) -> None:
        provider_model = getattr(embedding_provider, "model_name", None)
        if provider_model != store.embedding_model:
            raise ValueError("embedding provider model does not match database")
        if embedding_provider.dimension != store.embedding_dimension:
            raise ValueError("embedding provider dimension does not match database")
        if preview_length <= 0:
            raise ValueError("preview_length must be greater than zero")
        self.store = store
        self.embedding_provider = embedding_provider
        self.preview_length = preview_length

    def search(self, query: str, top_k: int = 3) -> List[PostgresSearchResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        query_embedding = self.embedding_provider.embed_text(query)
        rows = self.store.search(query_embedding, top_k)
        return [
            PostgresSearchResult(
                rank=row["rank"],
                similarity_score=float(row["similarity_score"]),
                document_title=row["document_title"],
                source_filename=row["source_filename"],
                section_title=row["section_title"],
                chunk_index=row["chunk_index"],
                chunk_text_preview=row["chunk_text"][: self.preview_length],
            )
            for row in rows
        ]
