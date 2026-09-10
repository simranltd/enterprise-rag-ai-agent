"""Small, swappable embedding providers for educational semantic search."""

from __future__ import annotations

from hashlib import sha256
import math
import re
from typing import List, Protocol, Sequence

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "for",
    "if",
    "i",
    "is",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "with",
}


class EmbeddingProvider(Protocol):
    """Interface implemented by local and future hosted embedding providers."""

    dimension: int

    def embed_text(self, text: str) -> List[float]:
        ...

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class LocalHashEmbeddingProvider:
    """Deterministic, offline embeddings using word and character features.

    This is intentionally not a replacement for a trained language model. It
    provides a transparent local baseline while keeping the provider swappable.
    Character n-grams help related word forms share some vector dimensions.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        raw_tokens = re.findall(r"[A-Za-z0-9]+", text)
        tokens = [token.lower() for token in raw_tokens if token.lower() not in _STOPWORDS]
        if not tokens:
            return vector

        features = [(token, 8.0) for token in tokens]
        features.extend(
            (f"acronym:{token.lower()}", 20.0)
            for token in raw_tokens
            if token.isupper() and len(token) > 1
        )
        features.extend(
            (f"char:{token[index:index + 3]}", 1.0)
            for token in tokens
            for index in range(max(1, len(token) - 2))
        )
        for feature, weight in features:
            digest = sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]
