"""Sentence-transformers embedding provider for Phase 1C."""

from __future__ import annotations

from typing import Any, List, Sequence


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


class NeuralEmbeddingProvider:
    """Adapter around a local SentenceTransformer model.

    The model name is configurable so the provider can be evaluated with
    another compatible local model without changing the search component.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for NeuralEmbeddingProvider"
                ) from error
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            get_dimension = getattr(self.model, "get_embedding_dimension", None)
            if get_dimension is None:
                get_dimension = self.model.get_sentence_embedding_dimension
            self._dimension = int(get_dimension())
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return self._encode(list(texts))

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
