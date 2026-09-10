"""Embedding provider abstractions and local implementations."""

from .provider import EmbeddingProvider, LocalHashEmbeddingProvider
from .neural_provider import (
    DEFAULT_MODEL_NAME,
    EXPECTED_DIMENSION,
    NeuralEmbeddingProvider,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EXPECTED_DIMENSION",
    "EmbeddingProvider",
    "LocalHashEmbeddingProvider",
    "NeuralEmbeddingProvider",
]
