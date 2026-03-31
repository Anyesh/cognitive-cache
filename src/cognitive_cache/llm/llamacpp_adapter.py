"""llama.cpp adapter via OpenAI-compatible API.

Works with any llama.cpp server (or compatible) that exposes /v1/chat/completions.
Zero cost, unlimited runs — perfect for benchmark iteration.
"""

import json
import os
import urllib.request

from cognitive_cache.llm.adapter import LLMAdapter, LLMResponse

LLAMACPP_DEFAULT_URL = "http://localhost:8080"
LLAMACPP_DEFAULT_MODEL = "Qwen3.5-9B-Q4_K_M"


class LlamaCppAdapter(LLMAdapter):
    """Adapter for local llama.cpp server with OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self._base_url = (
            base_url or os.environ.get("LLAMACPP_BASE_URL") or LLAMACPP_DEFAULT_URL
        ).rstrip("/")
        self._model = (
            model or os.environ.get("LLAMACPP_MODEL") or LLAMACPP_DEFAULT_MODEL
        )

    def complete(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0
    ) -> str:
        resp = self.complete_with_metadata(prompt, max_tokens, temperature)
        return resp.content

    def complete_with_metadata(
        self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0
    ) -> LLMResponse:
        payload = json.dumps(
            {
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()

        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())

        message = data["choices"][0]["message"]
        content = message.get("content", "") or ""

        # Qwen 3.5 models use reasoning_content for chain-of-thought.
        # If content is empty, the model spent all tokens thinking.
        # We combine both so we don't lose the output.
        reasoning = message.get("reasoning_content", "") or ""
        if not content.strip() and reasoning.strip():
            content = reasoning

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
