"""Deterministic fake provider for orchestration tests and demonstrations."""

from __future__ import annotations

from .base import LLMResponse


class FakeLLMProvider:
    """Records prompts and returns transparent, non-intelligent test output."""

    provider = "fake"
    model = "fake-deterministic-v1"

    def __init__(self, response_text: str = "[FAKE TEST OUTPUT]") -> None:
        self.response_text = response_text
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(
            text=self.response_text,
            provider=self.provider,
            model=self.model,
        )
