"""Model-agnostic LLM interface.

All LLM calls in the benchmark go through this interface.
This ensures we test the same prompt with the same parameters
across different models — the only variable is the model itself.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM API call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMAdapter(ABC):
    """Abstract interface for LLM API calls."""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
        ...

    @abstractmethod
    def complete_with_metadata(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0
    ) -> LLMResponse:
        ...
