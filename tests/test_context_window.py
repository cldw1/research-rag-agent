from dataclasses import replace

import pytest

from src.agent.context_window import ByteTokenEstimator, SlidingWindow, validate_turn
from src.llm.provider import ModelConfig, ModelError


def exchange(text="evidence"):
    return [{"role": "user", "content": "question"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": text},
            {"role": "assistant", "content": "answer [1]"}]


def test_window_drops_whole_old_turn_including_assistant_tool_pairs():
    config = ModelConfig(context_window=1800, max_output_tokens=128, safety_margin=128)
    old = exchange("old evidence " * 100)
    recent = [{"role": "user", "content": "recent"}, {"role": "assistant", "content": "recent answer"}]
    current = [{"role": "user", "content": "new question"}]
    chosen = SlidingWindow(config).select("system", [old, recent], current)
    assert chosen.dropped_turns == 1
    assert [m["role"] for m in chosen.messages] == ["system", "user", "assistant", "user"]
    assert chosen.input_estimate <= chosen.input_budget
    assert old[2]["content"].startswith("old evidence")


def test_current_tool_exchange_is_never_split_even_if_budget_cannot_fit():
    config = ModelConfig(context_window=1024, max_output_tokens=128, safety_margin=128)
    current = exchange("new evidence " * 200)[:-1]
    with pytest.raises(ModelError, match="当前问题、工具和证据"):
        SlidingWindow(config).select("system", [], current)


def test_schemas_and_evidence_count_toward_budget_and_output_is_reserved():
    counter = ByteTokenEstimator()
    messages = [{"role": "user", "content": "中文问题"}]
    assert counter.count(messages, [{"parameters": "schema " * 50}]) > counter.count(messages)
    config = ModelConfig(context_window=1200, max_output_tokens=200, safety_margin=128)
    result = SlidingWindow(config).select("system", [], messages)
    assert result.input_budget == 872
    assert "estimate" in result.count_method


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "q"}, {"role": "tool", "tool_call_id": "unknown", "content": "x"}],
    exchange()[:2] + [{"role": "assistant", "content": "unpaired"}],
    exchange()[:3],
    [{"role": "system", "content": "not a user turn"}],
])
def test_invalid_or_incomplete_tool_history_is_rejected(messages):
    with pytest.raises(ModelError, match="回合|工具|配对"):
        validate_turn(messages)


def test_complete_exchange_remains_valid_after_selection():
    chosen = SlidingWindow(ModelConfig()).select("system", [exchange()], [{"role": "user", "content": "next"}])
    validate_turn(chosen.messages[1:-1])
