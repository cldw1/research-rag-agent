"""Show the existing sliding window with fixed messages, without models or APIs."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.context_window import ByteTokenEstimator, SlidingWindow, validate_turn
from src.agent.tool_schema import TOOLS
from src.llm.provider import ModelConfig, ModelError


def demonstrate():
    config = ModelConfig(context_window=4096, max_output_tokens=256, safety_margin=128)
    system = "Use research evidence and preserve source citations."
    old = [{"role": "user", "content": "Earlier research question"},
           {"role": "assistant", "content": None, "tool_calls": [{"id": "demo_search_1", "type": "function",
                "function": {"name": "search_research_paper", "arguments": '{"paper_id":"demo","question":"evidence"}'}}]},
           {"role": "tool", "tool_call_id": "demo_search_1", "content": "Earlier retrieved evidence. " * 200},
           {"role": "assistant", "content": "Earlier answer [1]."}]
    recent = [{"role": "user", "content": "Recent follow-up question"},
              {"role": "assistant", "content": "Recent answer."}]
    current = [{"role": "user", "content": "Current question"}]
    history, baseline = [old, recent], deepcopy([old, recent])
    counter = ByteTokenEstimator()
    before = counter.count([{"role": "system", "content": system}] + old + recent + current, TOOLS)
    selected = SlidingWindow(config).select(system, history, current, TOOLS)
    assert selected.dropped_turns == 1 and history == baseline
    assert selected.input_estimate <= selected.input_budget
    assert all(message["role"] != "tool" and not message.get("tool_calls") for message in selected.messages)
    larger = SlidingWindow(ModelConfig(context_window=20000, max_output_tokens=256, safety_margin=128)).select(
        system, history, current, TOOLS)
    assert larger.dropped_turns == 0
    validate_turn(larger.messages[1:1 + len(old)])
    ongoing = deepcopy(old[:-1])
    ongoing[-1]["content"] = "Current retrieved evidence. " * 500
    try:
        SlidingWindow(config).select(system, [], ongoing, TOOLS)
    except ModelError as error:
        assert error.code == "context_budget_exceeded"
        oversized_code = error.code
    else:
        raise AssertionError("Oversized current evidence must not be silently truncated")
    return {"fixture": "two fixed synthetic history turns and one current question",
        "model_calls": 0, "models_loaded": 0, "count_method": selected.count_method,
        "application_context_budget": config.context_window,
        "reserved_output": config.max_output_tokens, "safety_margin": config.safety_margin,
        "input_budget": selected.input_budget, "before_estimate": before,
        "after_estimate": selected.input_estimate, "history_turns": len(history),
        "dropped_history_turns": selected.dropped_turns,
        "kept_message_roles": [message["role"] for message in selected.messages],
        "source_history_unchanged": history == baseline,
        "larger_budget_keeps_whole_tool_pair": larger.dropped_turns == 0,
        "oversized_current_error": oversized_code,
        "limits": "Serialized-byte estimate, not model tokens or API billing; no database operations in this demo."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output; an existing file is never overwritten")
    args = parser.parse_args()
    result = demonstrate()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
