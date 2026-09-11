"""Explicit retrieval, context, prompt, and provider orchestration."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, Protocol

from northstar.llm import LLMProvider

from .context_builder import ContextBuilder
from .models import RAGResult
from .prompt_builder import INSUFFICIENT_EVIDENCE_RESPONSE, PromptBuilder


class EmbeddingProviderLike(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...


class RetrieverLike(Protocol):
    def search(self, query_embedding: list[float], top_k: int = 3) -> Iterable[Any]:
        ...


class RAGOrchestrator:
    """Coordinates retrieval and future LLM generation through injection."""

    def __init__(
        self,
        embedding_provider: EmbeddingProviderLike,
        retriever: RetrieverLike,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        top_k: int = 3,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.embedding_provider = embedding_provider
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.top_k = top_k

    def answer(self, question: str) -> RAGResult:
        if not question.strip():
            raise ValueError("question cannot be empty")
        request_started = perf_counter()

        embedding_started = perf_counter()
        query_embedding = self.embedding_provider.embed_text(question)
        print(f"[RAG timing] query embedding: {perf_counter() - embedding_started:.3f}s")

        retrieval_started = perf_counter()
        retrieval_results = list(
            self.retriever.search(query_embedding, top_k=self.top_k)
        )
        print(f"[RAG timing] Chroma retrieval: {perf_counter() - retrieval_started:.3f}s")

        context_started = perf_counter()
        context = self.context_builder.build(retrieval_results)
        if not context.sources:
            print(
                "[RAG timing] context/prompt construction: "
                f"{perf_counter() - context_started:.3f}s"
            )
            print(
                f"[RAG timing] total request: {perf_counter() - request_started:.3f}s"
            )
            return RAGResult(
                question=question,
                answer_text=INSUFFICIENT_EVIDENCE_RESPONSE,
                sources=[],
                provider="none",
                model="none",
            )
        prompt = self.prompt_builder.build(question, context)
        prompt = self.prompt_builder.build(question, context)
        print(
            "[RAG timing] context/prompt construction: "
            f"{perf_counter() - context_started:.3f}s"
        )

        generation_started = perf_counter()
        response = self.llm_provider.generate(prompt)
        print(f"[RAG timing] LLM generation: {perf_counter() - generation_started:.3f}s")
        print(f"[RAG timing] total request: {perf_counter() - request_started:.3f}s")
        return RAGResult(
            question=question,
            answer_text=response.text,
            sources=context.sources,
            provider=response.provider,
            model=response.model,
        )
