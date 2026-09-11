"""Provider-independent language model interfaces."""

from .base import LLMProvider, LLMResponse
from .fake_provider import FakeLLMProvider

__all__ = ["LLMProvider", "LLMResponse", "FakeLLMProvider"]
