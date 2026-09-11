"""Persistent local Chroma storage for application-generated embeddings."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import chromadb

from northstar.ingestion import Chunk


DEFAULT_PERSISTENCE_DIRECTORY = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "northstar_document_chunks"


def content_hash(chunk_text: str) -> str:
    return sha256(chunk_text.encode("utf-8")).hexdigest()


def deterministic_chunk_id(chunk: Chunk) -> str:
    metadata = chunk.metadata
    identity = "|".join(
        (
            str(metadata.get("source_filename", "")),
            str(metadata.get("document_version", "")),
            str(metadata.get("chunk_index", "")),
            content_hash(chunk.text),
        )
    )
    return sha256(identity.encode("utf-8")).hexdigest()


class ChromaVectorStore:
    """Persistent Chroma collection using only embeddings supplied by callers."""

    def __init__(
        self,
        persistence_directory: str | Path = DEFAULT_PERSISTENCE_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.persistence_directory = Path(persistence_directory)
        self.client = chromadb.PersistentClient(path=str(self.persistence_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    @property
    def collection_name(self) -> str:
        return self.collection.name

    def upsert_chunks(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[List[float]],
    ) -> int:
        chunks_list = list(chunks)
        embeddings_list = list(embeddings)
        if len(chunks_list) != len(embeddings_list):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks_list:
            return 0

        ids = [deterministic_chunk_id(chunk) for chunk in chunks_list]
        self.collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks_list],
            embeddings=embeddings_list,
            metadatas=[self._metadata(chunk) for chunk in chunks_list],
        )
        return len(chunks_list)

    def count(self) -> int:
        return self.collection.count()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Mapping[str, Any]]:
        if top_k <= 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {
                "rank": rank,
                "chunk_text": document,
                "metadata": metadata,
                "distance": float(distance),
            }
            for rank, (document, metadata, distance) in enumerate(
                zip(documents, metadatas, distances), start=1
            )
        ]

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, Any]:
        source = chunk.metadata
        return {
            "document_title": str(source.get("document_title", "")),
            "source_filename": str(source.get("source_filename", "")),
            "document_version": str(source.get("document_version", "")),
            "document_type": str(source.get("document_type", "")),
            "department": str(source.get("department", "")),
            "jurisdiction": str(source.get("jurisdiction", "")),
            "classification": str(source.get("classification", "")),
            "section_title": str(source.get("section_title", "")),
            "chunk_index": int(source.get("chunk_index", 0)),
            "embedding_model": str(source.get("embedding_model", "")),
            "embedding_dimension": int(source.get("embedding_dimension", 0)),
            "content_hash": content_hash(chunk.text),
        }
