import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx2
import pytest

from src.llm.factory import CATALOG, create_provider
from src.llm.native import AnthropicProvider, GeminiProvider, OpenAIResponsesProvider
from src.llm.provider import ModelConfig, ModelError
from src.agent.tool_schema import TOOLS
from tests.test_llm_provider import public_resolver


def config(name, **kwargs):
    item = CATALOG[name]
    return ModelConfig(provider=item["provider"], model=item["model"], base_url=item["base_url"], **kwargs)


def normalized_call(completion):
    return {"role": "assistant", "content": completion.content,
            "tool_calls": [call.to_message() for call in completion.tool_calls],
            "provider_state": completion.provider_state}


class ClosingEvents:
    def __init__(self, events):
        self.events, self.closed = events, False
    def __iter__(self):
        return iter(self.events)
    def close(self):
        self.closed = True


def sse(events):
    return b"".join(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n" for event in events)


def test_openai_responses_preserves_reasoning_phase_call_id_and_store_false():
    payload = {"status": "completed", "output": [
        {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": "opaque-test-state"},
        {"type": "function_call", "id": "fc1", "call_id": "call1", "name": "get_paper_outline", "arguments": '{"paper_id":"p"}'}],
        "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}
    client = Mock()
    client.responses.create.return_value = payload
    provider = OpenAIResponsesProvider(config("OpenAI Responses"), "dummy", client=client, resolver=public_resolver)
    response = provider.complete([{"role": "system", "content": "system"}, {"role": "user", "content": "q"}], tools=TOOLS, tool_choice="required")
    assert response.tool_calls[0].id == "call1"
    messages = [{"role": "user", "content": "q"}, normalized_call(response), {"role": "tool", "tool_call_id": "call1", "content": "result"}]
    provider.complete(messages, tools=TOOLS)
    request = client.responses.create.call_args.kwargs
    assert request["store"] is False and "temperature" not in request
    assert request["input"][-1] == {"type": "function_call_output", "call_id": "call1", "output": "result"}
    assert request["input"][1:3] == payload["output"]


def test_openai_stream_uses_completed_response_instead_of_partial_reasoning_items():
    final = {"status": "completed", "output": [{"type": "message", "id": "m1", "role": "assistant", "phase": "final_answer",
              "content": [{"type": "output_text", "text": "Hello"}]}]}
    events = ClosingEvents([{"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"}, {"type": "response.completed", "response": final}])
    client = Mock()
    client.responses.create.return_value = events
    provider = OpenAIResponsesProvider(config("OpenAI Responses", stream=True), "dummy", client=client, resolver=public_resolver)
    result = list(provider.stream_complete([{"role": "user", "content": "q"}]))
    assert "".join(event.text for event in result if event.kind == "text_delta") == "Hello"
    assert result[-1].completion.provider_state["content"] == final["output"] and events.closed


def test_claude_native_tool_results_signatures_and_sampling_are_correct():
    blocks = [{"type": "thinking", "thinking": "fixture", "signature": "opaque-signature"},
              {"type": "tool_use", "id": "c1", "name": "get_paper_outline", "input": {"paper_id": "p"}}]
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        assert request.headers["x-api-key"] == "dummy"
        return httpx2.Response(200, json={"content": blocks, "stop_reason": "tool_use", "usage": {"input_tokens": 3, "cache_read_input_tokens": 7, "output_tokens": 2}})
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        provider = AnthropicProvider(config("Anthropic Claude"), "dummy", client=client, resolver=public_resolver)
        first = provider.complete([{"role": "user", "content": "q"}], tools=TOOLS, tool_choice="required")
        provider.complete([{"role": "user", "content": "q"}, normalized_call(first),
            {"role": "tool", "tool_call_id": "c1", "content": "evidence"}], tools=TOOLS)
    request = requests[-1]
    assert request["messages"][1]["content"] == blocks
    assert request["messages"][-1]["content"][0]["tool_use_id"] == "c1"
    assert request["tool_choice"] == {"type": "auto"} and "temperature" not in request
    assert first.usage["prompt_tokens"] == 10 and first.usage["total_tokens"] == 12


def test_claude_stream_assembles_partial_json_and_thinking_signatures():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "opaque"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "c1", "name": "get_paper_outline", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"paper_'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'id":"p"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 6}}, {"type": "message_stop"}]
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, content=sse(events)))) as client:
        provider = AnthropicProvider(config("Anthropic Claude", stream=True), "dummy", client=client, resolver=public_resolver)
        result = list(provider.stream_complete([{"role": "user", "content": "q"}], tools=TOOLS))[-1].completion
    assert json.loads(result.tool_calls[0].arguments) == {"paper_id": "p"}
    assert result.provider_state["content"][0]["signature"] == "opaque"


def test_gemini_keeps_original_parts_and_parallel_function_response_ids():
    parts = [{"functionCall": {"id": "c1", "name": "get_paper_outline", "args": {"paper_id": "p"}}, "thoughtSignature": "sig1"},
             {"functionCall": {"id": "c2", "name": "get_paper_outline", "args": {"paper_id": "p"}}}]
    payload = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        assert "key=" not in str(request.url) and request.headers["x-goog-api-key"] == "dummy"
        return httpx2.Response(200, json=payload)
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        provider = GeminiProvider(config("Google Gemini"), "dummy", client=client, resolver=public_resolver)
        first = provider.complete([{"role": "user", "content": "q"}], tools=TOOLS)
        provider.complete([{"role": "user", "content": "q"}, normalized_call(first),
            {"role": "tool", "tool_call_id": "c1", "content": "result1"},
            {"role": "tool", "tool_call_id": "c2", "content": "result2"}], tools=TOOLS)
    assert requests[-1]["contents"][1]["parts"] == parts
    assert [part["functionResponse"]["id"] for part in requests[-1]["contents"][-1]["parts"]] == ["c1", "c2"]
    assert requests[-1]["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]


def test_gemini_stream_preserves_empty_text_signature_without_merging():
    contents = [{"role": "model", "parts": [{"text": "中文"}]}, {"role": "model", "parts": [{"text": "", "thoughtSignature": "signature"}]}]
    events = [{"candidates": [{"content": contents[0]}]},
              {"candidates": [{"content": contents[1], "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7}}]
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, content=sse(events)))) as client:
        provider = GeminiProvider(config("Google Gemini", stream=True), "dummy", client=client, resolver=public_resolver)
        result = list(provider.stream_complete([{"role": "user", "content": "q"}]))
    assert result[-1].completion.content == "中文"
    assert result[-1].completion.provider_state["content"] == contents


@pytest.mark.parametrize("name", ["Anthropic Claude", "Google Gemini"])
def test_native_stream_disconnect_and_authentication_do_not_leak_credentials(name):
    credential = "do-not-persist-this-credential"
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(401, json={"error": credential}))) as client:
        provider = create_provider(config(name), credential, client=client, resolver=public_resolver)
        with pytest.raises(ModelError) as error:
            provider.complete([{"role": "user", "content": "q"}])
        assert error.value.code == "authentication_failed" and credential not in str(error.value)
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, content=b'data: {}\n\n'))) as client:
        provider = create_provider(config(name, stream=True), "dummy", client=client, resolver=public_resolver)
        with pytest.raises(ModelError, match="中断"):
            list(provider.stream_complete([{"role": "user", "content": "q"}]))


def test_bailian_uses_its_own_non_thinking_extension_and_validates_workspace_url():
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="answer", tool_calls=[]), finish_reason="stop")], usage=None)
    provider = create_provider(config("阿里云百炼"), "dummy", client=client, resolver=public_resolver)
    provider.complete([{"role": "user", "content": "q"}])
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {"enable_thinking": False}
    with pytest.raises(ModelError):
        replace(config("阿里云百炼"), base_url="https://example.com/compatible-mode/v1")


def test_claude_retains_definitions_with_tool_choice_none_at_budget_limit():
    provider = AnthropicProvider(config("Anthropic Claude"), "dummy", client=Mock(), resolver=public_resolver)
    request = provider._request([{"role": "user", "content": "q"}], TOOLS, "none")
    assert request["tools"] and request["tool_choice"] == {"type": "none"}


def test_sse_decodes_utf8_even_when_every_byte_is_a_separate_read():
    from src.llm.native import sse_json
    raw = sse([{"text": "中文签名"}])
    class ByteResponse:
        def iter_bytes(self, chunk_size):
            yield from (raw[i:i + 1] for i in range(len(raw)))
    assert list(sse_json(ByteResponse(), None)) == [{"text": "中文签名"}]


@pytest.mark.parametrize("name", ["Anthropic Claude", "Google Gemini"])
def test_native_transport_retries_once_only_before_a_nonstream_response(name):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx2.ConnectError("fixture secret must not appear", request=request)
        if name == "Anthropic Claude":
            return httpx2.Response(200, json={"content": [{"type": "text", "text": "answer"}], "stop_reason": "end_turn"})
        return httpx2.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "answer"}]}, "finishReason": "STOP"}]})
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        provider = create_provider(config(name), "dummy", client=client, resolver=public_resolver)
        assert provider.complete([{"role": "user", "content": "q"}]).content == "answer"
    assert len(calls) == 2
