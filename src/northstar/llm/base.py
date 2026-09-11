"""Small provider-independent LLM abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMProvider(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response from a fully constructed prompt."""
        ...
