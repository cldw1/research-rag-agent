"""Trace the actual ResearchAgent with scripted model replies and one local chunk.

No LLM API, embedding weights or business database are used. This demonstrates
control flow; the scripted decisions are not a model capability evaluation.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.context_window import validate_turn
from src.agent.research_agent import ResearchAgent
from src.llm.provider import Completion, ModelConfig, ToolCall


class FixtureRetriever:
    paper_id = "demo_paper"
    has_verified_sections = False

    def __init__(self):
        self.search_count = 0
        self.chunks = [{"paper_id": self.paper_id, "chunk_id": "demo_chunk_1",
            "title": "Synthetic research example", "section_name": "Methods",
            "section_index": 0, "paragraph_index": 0, "piece_index": 0,
            "text": "The fixed validation dataset measures retrieval recall. Answers must preserve sources."}]

    def get_paper_data(self, paper_id):
        if paper_id != self.paper_id:
            raise ValueError("Fixture only permits the selected document")
        return {"chunks": deepcopy(self.chunks)}

    def search(self, query, paper_id, **kwargs):
        self.get_paper_data(paper_id)
        self.search_count += 1
        return [{**self.chunks[0], "rank": 1}]  # Fixed fixture, no retrieval-quality claim.


class ScriptedProvider:
    def __init__(self):
        self.config, self.requests = ModelConfig(max_output_tokens=256), []

    def complete(self, messages, *, tools=None, tool_choice=None):
        self.requests.append({"request": len(self.requests) + 1,
            "message_roles": [message["role"] for message in messages],
            "tools_available": bool(tools), "tool_choice": tool_choice})
        if len(self.requests) == 1:
            return Completion(tool_calls=(ToolCall("outline_1", "get_paper_outline", '{"paper_id":"demo_paper"}'),))
        if len(self.requests) == 2:
            assert messages[-1]["tool_call_id"] == "outline_1"
            return Completion(tool_calls=(ToolCall("search_1", "search_research_paper",
                '{"paper_id":"demo_paper","question":"fixed validation dataset"}'),))
        if len(self.requests) == 3:
            assert messages[-1]["tool_call_id"] == "search_1" and not tools
            assert "measures retrieval recall" in messages[-1]["content"]
            return Completion("固定验证集用于检查检索召回，回答需要保留来源 [1]。", finish_reason="stop")
        raise AssertionError("The completed fixture must stop after the third model response")


def demonstrate():
    provider, retriever, events = ScriptedProvider(), FixtureRetriever(), []
    agent = ResearchAgent(provider, retriever)
    result = agent.run("这篇文档怎样使用验证集？", on_event=events.append)
    validate_turn(result["messages"])
    requested = [call["id"] for message in result["messages"] for call in message.get("tool_calls", [])]
    returned = [message["tool_call_id"] for message in result["messages"] if message["role"] == "tool"]
    assert requested == returned == ["outline_1", "search_1"]
    assert result["model_calls"] == 3 and retriever.search_count == 1
    return {"purpose": "actual Agent control flow with fixed model/retrieval fixtures",
        "llm_api_calls": 0, "embedding_models_loaded": 0, "database_operations": 0,
        "requests": provider.requests, "events": events,
        "tool_trace": result["tool_calls"], "message_roles": [message["role"] for message in result["messages"]],
        "requested_call_ids": requested, "returned_call_ids": returned,
        "scripted_model_responses": result["model_calls"], "executed_tools": len(result["tool_calls"]),
        "retrieval_fixture_calls": retriever.search_count, "answer": result["answer"],
        "cited_numbers": result["cited_numbers"], "invalid_citations": result["invalid_citations"],
        "limits": "No real model decision-making, retrieval ranking, database persistence or answer-quality verification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report; existing files are never overwritten")
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
