"""OpenAI API adapter. Requires OPENAI_API_KEY env var."""

import os

from cognitive_cache.llm.adapter import LLMAdapter, LLMResponse


class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str = "gpt-4o"):
        try:
            import openai
        except ImportError:
            raise ImportError("Install openai: pip install cognitive-cache[benchmark]")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def complete_with_metadata(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self._model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
