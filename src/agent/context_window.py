"""Budget complete turns; preserve all tool-call/result pairs in current work."""

import json
from copy import deepcopy
from dataclasses import dataclass

from src.llm.provider import ModelError


class ByteTokenEstimator:
    method = "utf8_serialized_bytes_plus_framing_estimate_v1"

    def count(self, messages, tools=None):
        payload = {"messages": messages, "tools": tools or []}
        # Includes schemas, arguments, results, evidence and message metadata.
        # Deliberately labeled an estimate; not the provider's tokenizer/billing.
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 16 * len(messages) + 32


def validate_turn(messages, *, complete=True):
    if not messages or messages[0].get("role") != "user":
        raise ModelError("invalid_history", "对话回合必须从用户消息开始")
    pending, seen = set(), set()
    for position, message in enumerate(messages):
        role = message.get("role")
        if position == 0:
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ModelError("invalid_history", "发现没有配对请求的工具结果")
            pending.remove(call_id)
        elif role == "assistant":
            if pending:
                raise ModelError("invalid_history", "工具请求缺少对应结果")
            calls = message.get("tool_calls") or []
            for call in calls:
                if not call.get("id") or call["id"] in seen:
                    raise ModelError("invalid_history", "工具调用 ID 缺失或重复")
                pending.add(call["id"])
                seen.add(call["id"])
        else:
            raise ModelError("invalid_history", "回合中存在不支持的消息角色")
    if pending or (complete and (messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"))):
        raise ModelError("invalid_history", "对话回合或工具结果尚未完成")


@dataclass(frozen=True)
class ContextSelection:
    messages: list[dict]
    input_estimate: int
    input_budget: int
    dropped_turns: int
    count_method: str


class SlidingWindow:
    def __init__(self, config, counter=None):
        self.config = config
        self.counter = counter or ByteTokenEstimator()

    def select(self, system, history, current, tools=None):
        validate_turn(current, complete=False)
        for turn in history:
            validate_turn(turn)
        budget = self.config.context_window - self.config.max_output_tokens - self.config.safety_margin
        base = [{"role": "system", "content": system}]
        if self.counter.count(base + current, tools) > budget:
            raise ModelError("context_budget_exceeded", "当前问题、工具和证据已超出输入预算，请增加预算或缩小文档分块/证据范围")
        start = 0
        while True:
            messages = base + [message for turn in history[start:] for message in turn] + current
            count = self.counter.count(messages, tools)
            if count <= budget:
                return ContextSelection(deepcopy(messages), count, budget, start, self.counter.method)
            start += 1  # Only whole completed turns are removed; database stays intact.
