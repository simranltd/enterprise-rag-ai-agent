"""Deterministic evidence responses for fast local knowledge requests."""

from __future__ import annotations

from .context_builder import ContextBuildResult
from .models import RAGResult
from .prompt_builder import INSUFFICIENT_EVIDENCE_RESPONSE


class KnowledgeResponseBuilder:
    """Render retrieved current-policy evidence without invoking an LLM."""

    provider = "deterministic-evidence"
    model = "none"

    def build(self, question: str, context: ContextBuildResult) -> RAGResult:
        if not context.sources:
            return RAGResult(
                question=question,
                answer_text=INSUFFICIENT_EVIDENCE_RESPONSE,
                sources=[],
                provider=self.provider,
                model=self.model,
            )

        usable_sources = [
            source for source in context.sources if source.chunk_text.strip()
        ]
        evidence = "\n\n".join(
            f"{source.citation_id} {source.chunk_text.strip()}"
            for source in usable_sources
        )
        if not evidence:
            return RAGResult(
                question=question,
                answer_text=INSUFFICIENT_EVIDENCE_RESPONSE,
                sources=[],
                provider=self.provider,
                model=self.model,
            )
        return RAGResult(
            question=question,
            answer_text=evidence,
            sources=usable_sources,
            provider=self.provider,
            model=self.model,
        )
