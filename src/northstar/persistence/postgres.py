"""Thin PostgreSQL helper with visible SQL for the Phase 1D learning path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from northstar.ingestion import Chunk


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT / "src" / "northstar" / "persistence" / "schema.sql"


class PostgresChunkStore:
    """Stores and searches 384-dimensional chunk embeddings in PostgreSQL."""

    def __init__(
        self,
        connection_string: str,
        embedding_model: str,
        embedding_dimension: int = 384,
    ) -> None:
        if not connection_string.strip():
            raise ValueError("connection_string cannot be empty")
        if not embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        if embedding_dimension != 384:
            raise ValueError("document_chunks requires embedding_dimension=384")
        self.connection_string = connection_string
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

    def connect(self) -> psycopg.Connection:
        connection = psycopg.connect(self.connection_string, row_factory=dict_row)
        register_vector(connection)
        return connection

    def initialize_schema(self) -> None:
        with self.connect() as connection:
            statements = [
                statement.strip()
                for statement in SCHEMA_PATH.read_text(encoding="utf-8").split(";")
                if statement.strip()
            ]
            for statement in statements:
                connection.execute(statement)

    def insert_chunks(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[List[float]],
    ) -> int:
        rows = [
            self._chunk_row(chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        if not rows:
            return 0
        with self.connect() as connection:
            inserted = 0
            for row in rows:
                cursor = connection.execute(
                    """
                    INSERT INTO document_chunks (
                        document_title, source_filename, document_version,
                        document_type, department, jurisdiction, classification,
                        section_title, chunk_index, chunk_text, embedding_model,
                        embedding_dimension, embedding, content_hash
                    )
                    VALUES (
                        %(document_title)s, %(source_filename)s,
                        %(document_version)s, %(document_type)s,
                        %(department)s, %(jurisdiction)s, %(classification)s,
                        %(section_title)s, %(chunk_index)s, %(chunk_text)s,
                        %(embedding_model)s, %(embedding_dimension)s,
                        %(embedding)s, %(content_hash)s
                    )
                    ON CONFLICT (
                        source_filename, document_version, chunk_index, content_hash
                    ) DO NOTHING
                    """,
                    row,
                )
                inserted += cursor.rowcount
        return inserted

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Mapping[str, Any]]:
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"query embedding must have dimension {self.embedding_dimension}"
            )
        if top_k <= 0:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id, document_title, source_filename, document_version,
                    document_type, department, jurisdiction, classification,
                    section_title, chunk_index, chunk_text, embedding_model,
                    embedding_dimension,
                    1 - (embedding <=> %(query_embedding)s::vector)
                        AS similarity_score
                FROM document_chunks
                WHERE embedding_model = %(embedding_model)s
                  AND embedding_dimension = %(embedding_dimension)s
                ORDER BY embedding <=> %(query_embedding)s::vector
                LIMIT %(top_k)s
                """,
                {
                    "query_embedding": query_embedding,
                    "embedding_model": self.embedding_model,
                    "embedding_dimension": self.embedding_dimension,
                    "top_k": top_k,
                },
            ).fetchall()
        return [
            {
                "rank": rank,
                **dict(row),
            }
            for rank, row in enumerate(rows, start=1)
        ]

    def count(self) -> int:
        with self.connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) AS count FROM document_chunks"
            ).fetchone()["count"]

    def _chunk_row(self, chunk: Chunk, embedding: List[float]) -> dict[str, Any]:
        if len(embedding) != self.embedding_dimension:
            raise ValueError(
                f"embedding must have dimension {self.embedding_dimension}"
            )
        metadata = chunk.metadata
        import hashlib

        return {
            "document_title": str(metadata.get("document_title", "")),
            "source_filename": str(metadata.get("source_filename", "")),
            "document_version": str(metadata.get("document_version", "")),
            "document_type": str(metadata.get("document_type", "")),
            "department": str(metadata.get("department", "")),
            "jurisdiction": str(metadata.get("jurisdiction", "")),
            "classification": str(metadata.get("classification", "")),
            "section_title": str(metadata.get("section_title", "")),
            "chunk_index": int(metadata.get("chunk_index", 0)),
            "chunk_text": chunk.text,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "embedding": embedding,
            "content_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        }
