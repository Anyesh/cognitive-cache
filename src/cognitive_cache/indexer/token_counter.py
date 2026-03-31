"""Token counting using tiktoken (the tokenizer used by OpenAI/most LLM APIs).

Why tiktoken: It's the standard tokenizer for GPT models and gives a close
approximation for Claude as well. The exact count doesn't matter — we need
consistent relative sizing so the knapsack optimization makes fair comparisons.
"""

import tiktoken

# cl100k_base is the encoding used by GPT-4 and similar models.
# We load it once and reuse — it's expensive to initialize.
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    if not text:
        return 0
    return len(_encoder.encode(text))
