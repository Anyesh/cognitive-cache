import pytest

from cognitive_cache.llm.adapter import LLMAdapter, LLMResponse


def test_llm_response_structure():
    resp = LLMResponse(
        content="def fix(): pass",
        model="test-model",
        input_tokens=100,
        output_tokens=20,
    )
    assert resp.content == "def fix(): pass"
    assert resp.total_tokens == 120


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        LLMAdapter()
