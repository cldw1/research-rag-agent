"""Native wire formats. Signatures/reasoning items remain provider-scoped data."""

import json
import re
from copy import deepcopy

from src.llm.provider import (Completion, ModelError, ModelEvent, OpenAICompatibleProvider,
    ToolCall, check_public_target, message_text, validate_calls)
from src.llm.streaming import close_on_cancel


def native_state(config, content):
    return {"version": 1, "provider": config.provider, "model": config.model,
            "base_url": config.base_url.rstrip("/"), "content": content}


def own_state(message, config):
    state = message.get("provider_state") or {}
    if (state.get("version") == 1 and state.get("provider") == config.provider
            and state.get("model") == config.model and state.get("base_url") == config.base_url.rstrip("/")):
        return deepcopy(state.get("content"))
    return None


def normalize_usage(data, input_key="input_tokens", output_key="output_tokens", total_key="total_tokens"):
    result = {}
    for key, native in (("prompt_tokens", input_key), ("completion_tokens", output_key), ("total_tokens", total_key)):
        value = data.get(native)
        if isinstance(value, int) and value >= 0:
            result[key] = value
    if "total_tokens" not in result and "prompt_tokens" in result and "completion_tokens" in result:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def json_value(value):
    return value.model_dump(mode="json", exclude_none=True) if hasattr(value, "model_dump") else value


def _request_error(exc, token=None):
    if token:
        token.check()
    if isinstance(exc, ModelError):
        return exc
    import httpx2
    if isinstance(exc, httpx2.TimeoutException):
        return ModelError("model_timeout", "模型服务请求超时，请稍后重试")
    if isinstance(exc, httpx2.TransportError):
        return ModelError("model_connection", "无法完成模型连接，请检查网络")
    # Never include the upstream exception body or authenticated URL/headers.
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = exc.response.status_code
    code = "authentication_failed" if status in (401, 403) else "rate_limited" if status == 429 else "provider_error"
    return ModelError(code, "模型服务请求未完成，请检查授权、额度、网络和模型配置")


class OpenAIResponsesProvider(OpenAICompatibleProvider):
    def _request(self, messages, tools, tool_choice):
        items, instructions = [], []
        for message in messages:
            role = message["role"]
            if role == "system":
                instructions.append(message_text(message))
            elif role == "tool":
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": message_text(message)})
            elif role == "assistant":
                state = own_state(message, self.config)
                if state is not None:
                    items.extend(state)
                else:
                    if message.get("content"):
                        items.append({"role": "assistant", "content": message_text(message),
                                      "phase": "commentary" if message.get("tool_calls") else "final_answer"})
                    for call in message.get("tool_calls") or []:
                        items.append({"type": "function_call", "call_id": call["id"], **call["function"]})
            else:
                items.append({"role": "user", "content": message_text(message)})
        request = {"model": self.config.model, "instructions": "\n".join(instructions), "input": items,
                   "max_output_tokens": self.config.max_output_tokens, "store": False,
                   "include": ["reasoning.encrypted_content"]}
        # Sampling parameters are not universal across reasoning models.
        if tools:
            request["tools"] = [{"type": "function", **tool["function"], "strict": False} for tool in tools]
            request["tool_choice"] = tool_choice or "auto"
        return request

    def _decode(self, response):
        data = json_value(response)
        if data.get("status") not in ("completed", "incomplete"):
            raise ModelError("provider_error", "模型未完成当前响应")
        output = data.get("output") or []
        text, calls = [], []
        for item in output:
            if item["type"] == "message":
                for block in item.get("content", []):
                    if block["type"] == "output_text":
                        text.append(block["text"])
                    elif block["type"] == "refusal":
                        raise ModelError("model_refusal", "模型拒绝了当前请求")
            elif item["type"] == "function_call":
                calls.append(ToolCall(item["call_id"], item["name"], item["arguments"]))
            elif item["type"] != "reasoning":
                raise ModelError("unsupported_response", "模型返回了未启用的原生工具或内容类型")
        truncated = data.get("status") == "incomplete"
        if truncated and calls:
            raise ModelError("incomplete_tool_call", "模型工具请求被截断，未执行工具")
        return Completion("".join(text), validate_calls(calls), normalize_usage(data.get("usage") or {}),
                          "length" if truncated else "stop", native_state(self.config, output))

    def complete(self, messages, *, tools=None, tool_choice=None):
        check_public_target(self.config.base_url, self.resolver)
        try:
            return self._decode(self.client.responses.create(**self._request(messages, tools, tool_choice)))
        except Exception as exc:
            raise _request_error(exc) from None

    def stream_complete(self, messages, *, tools=None, tool_choice=None, cancellation=None):
        if cancellation:
            cancellation.check()
        check_public_target(self.config.base_url, self.resolver)
        try:
            stream = self.client.responses.create(**self._request(messages, tools, tool_choice), stream=True)
            terminal, size = None, 0
            with close_on_cancel(stream, cancellation):
                for event in stream:
                    if cancellation:
                        cancellation.check()
                    data = json_value(event)
                    if data["type"] == "response.output_text.delta":
                        delta = data["delta"]
                        size += len(delta)
                        if size > 2 * 1024 * 1024:
                            raise ModelError("response_too_large", "输出超过应用限制")
                        yield ModelEvent("text_delta", delta)
                    elif data["type"] in ("response.completed", "response.incomplete"):
                        terminal = data["response"]
                    elif data["type"] in ("error", "response.failed"):
                        raise ModelError("provider_error", "模型流返回失败事件")
            if cancellation:
                cancellation.check()
            if terminal is None:
                raise ModelError("incomplete_stream", "模型流中断，未保存部分回答")
            yield ModelEvent("completed", completion=self._decode(terminal))
        except Exception as exc:
            raise _request_error(exc, cancellation) from None


def sse_json(response, token):
    """Bound bytes before decoding, including a UTF-8 sequence split across reads."""
    buffer, lines, total, event_size = b"", [], 0, 0
    for chunk in response.iter_bytes(chunk_size=8192):
        if token:
            token.check()
        total += len(chunk)
        buffer += chunk
        if total > 8 * 1024 * 1024 or len(buffer) > 2 * 1024 * 1024:
            raise ModelError("response_too_large", "模型流超过应用大小限制")
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if line.startswith(b"data:"):
                line = line[5:].lstrip(b" ")
                lines.append(line)
                event_size += len(line)
                if event_size > 2 * 1024 * 1024:
                    raise ModelError("response_too_large", "单个流事件过大")
            elif not line and lines:
                raw = b"\n".join(lines)
                lines, event_size = [], 0
                yield {"_done": True} if raw == b"[DONE]" else json.loads(raw)
    # Missing terminal delimiters are not accepted as a completed response.


class NativeHTTPProvider:
    supports_required_tools = True

    def __init__(self, config, api_key, *, client=None, resolver=None):
        if not api_key or not api_key.strip():
            raise ModelError("missing_key", "请填写当前供应商的 API Key")
        self.config, self.resolver = config, resolver
        self._key, self._owns_client = api_key.strip(), client is None
        if client is None:
            from openai import DefaultHttpxClient
            client = DefaultHttpxClient(timeout=60.0, follow_redirects=False, trust_env=False)
        self.client = client

    def close(self):
        if self._owns_client:
            self.client.close()

    def _path(self, streaming):
        raise NotImplementedError

    def _headers(self):
        raise NotImplementedError

    def complete(self, messages, *, tools=None, tool_choice=None):
        check_public_target(self.config.base_url, self.resolver)
        try:
            request = self._request(messages, tools, tool_choice)
            # One bounded retry before a response is consumed; never replay a
            # partially received stream automatically.
            for attempt in range(2):
                try:
                    response = self.client.post(self._path(False), headers=self._headers(), json=request)
                except Exception as exc:
                    import httpx2
                    if isinstance(exc, httpx2.TransportError) and not attempt:
                        continue
                    raise
                if response.status_code not in (429, 502, 503, 504) or attempt:
                    break
                response.close()
            try:
                response.raise_for_status()
                return self._decode(response.json())
            finally:
                response.close()
        except Exception as exc:
            raise _request_error(exc) from None

    def stream_complete(self, messages, *, tools=None, tool_choice=None, cancellation=None):
        if cancellation:
            cancellation.check()
        check_public_target(self.config.base_url, self.resolver)
        try:
            request = self._request(messages, tools, tool_choice)
            if self.config.provider == "anthropic":
                request["stream"] = True
            with self.client.stream("POST", self._path(True), headers=self._headers(), json=request) as response:
                response.raise_for_status()
                with close_on_cancel(response, cancellation):
                    yield from self._accumulate(sse_json(response, cancellation), cancellation)
        except Exception as exc:
            raise _request_error(exc, cancellation) from None


def _append_content(messages, role, blocks):
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


class AnthropicProvider(NativeHTTPProvider):
    # Modern adaptive-thinking models reject forced tool use. Pre-grounding is
    # handled by the Agent; never silently send an unsupported tool_choice.
    supports_required_tools = False
    requires_tool_definitions_for_history = True

    def _path(self, streaming):
        return self.config.base_url.rstrip("/") + "/v1/messages"

    def _headers(self):
        return {"x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def _request(self, messages, tools, tool_choice):
        system, converted = [], []
        for message in messages:
            role = message["role"]
            if role == "system":
                system.append(message_text(message))
            elif role == "tool":
                _append_content(converted, "user", [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message_text(message)}])
            elif role == "assistant":
                blocks = own_state(message, self.config)
                if blocks is None:
                    blocks = [{"type": "text", "text": message_text(message)}] if message.get("content") else []
                    blocks += [{"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                                "input": json.loads(call["function"]["arguments"])} for call in message.get("tool_calls") or []]
                _append_content(converted, "assistant", blocks)
            else:
                _append_content(converted, "user", [{"type": "text", "text": message_text(message)}])
        request = {"model": self.config.model, "system": "\n".join(system), "messages": converted,
                   "max_tokens": self.config.max_output_tokens}
        if tools:
            request["tools"] = [{"name": tool["function"]["name"], "description": tool["function"].get("description", ""),
                                 "input_schema": tool["function"]["parameters"]} for tool in tools]
            request["tool_choice"] = {"type": "none" if tool_choice == "none" else "auto"}
        return request

    def _decode(self, data):
        if data.get("type") == "error":
            raise ModelError("provider_error", "Claude 返回失败事件")
        blocks, text, calls = data.get("content") or [], [], []
        for block in blocks:
            if block["type"] == "text":
                text.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], json.dumps(block["input"], ensure_ascii=False)))
            elif block["type"] not in ("thinking", "redacted_thinking"):
                raise ModelError("unsupported_response", "Claude 返回未启用的内容类型")
        reason = data.get("stop_reason")
        if reason not in ("end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal"):
            raise ModelError("incomplete_response", "Claude 未返回支持的结束状态")
        if reason == "max_tokens" and calls:
            raise ModelError("incomplete_tool_call", "Claude 工具参数被截断，未执行工具")
        receipt = data.get("usage") or {}
        tokens = dict(receipt)
        tokens["input_tokens"] = sum(receipt.get(name, 0) for name in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        normalized = normalize_usage(tokens)
        for name in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if isinstance(receipt.get(name), int):
                normalized[name] = receipt[name]
        return Completion("".join(text), validate_calls(calls), normalized,
                          "length" if reason == "max_tokens" else reason, native_state(self.config, blocks))

    def _accumulate(self, events, token):
        blocks, arguments, usage, reason, stopped, closed = {}, {}, {}, None, False, set()
        for data in events:
            kind = data.get("type")
            if kind == "error":
                raise ModelError("provider_error", "Claude 流返回失败事件")
            if kind == "message_start":
                usage.update(data["message"].get("usage") or {})
            elif kind == "content_block_start":
                blocks[data["index"]] = deepcopy(data["content_block"])
            elif kind == "content_block_delta":
                index, delta = data["index"], data["delta"]
                block = blocks[index]
                if delta["type"] == "text_delta":
                    block["text"] = block.get("text", "") + delta["text"]
                    yield ModelEvent("text_delta", delta["text"])
                elif delta["type"] == "input_json_delta":
                    arguments[index] = arguments.get(index, "") + delta["partial_json"]
                    if len(arguments[index]) > 16000:
                        raise ModelError("invalid_response", "工具参数超过应用限制")
                elif delta["type"] in ("thinking_delta", "signature_delta"):
                    key = "thinking" if delta["type"] == "thinking_delta" else "signature"
                    block[key] = block.get(key, "") + delta[key]
            elif kind == "content_block_stop":
                index = data["index"]
                if index in arguments:
                    blocks[index]["input"] = json.loads(arguments[index])
                closed.add(index)
            elif kind == "message_delta":
                reason = data.get("delta", {}).get("stop_reason", reason)
                usage.update(data.get("usage") or {})
            elif kind == "message_stop":
                stopped = True
        if token:
            token.check()
        if not stopped or set(blocks) != closed:
            raise ModelError("incomplete_stream", "Claude 流中断，未提交部分结果")
        yield ModelEvent("completed", completion=self._decode({"content": [blocks[i] for i in sorted(blocks)],
            "usage": usage, "stop_reason": reason}))


class GeminiProvider(NativeHTTPProvider):
    def _path(self, streaming):
        model = self.config.model.removeprefix("models/")
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", model):
            raise ModelError("invalid_model", "Gemini 模型名称无效")
        action = "streamGenerateContent?alt=sse" if streaming else "generateContent"
        return self.config.base_url.rstrip("/") + f"/models/{model}:{action}"

    def _headers(self):
        return {"x-goog-api-key": self._key, "content-type": "application/json"}

    def _request(self, messages, tools, tool_choice):
        system, contents, call_names = [], [], {}
        for message in messages:
            role = message["role"]
            if role == "system":
                system.append(message_text(message))
            elif role == "tool":
                name, native_id = call_names[message["tool_call_id"]]
                function = {"name": name, "response": {"result": message_text(message)}}
                if native_id:
                    function["id"] = native_id
                # All parallel responses follow all calls, in one user Content.
                part = {"functionResponse": function}
                if contents and contents[-1]["role"] == "user" and any("functionResponse" in p for p in contents[-1]["parts"]):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
            elif role == "assistant":
                state = own_state(message, self.config)
                if state is not None:
                    contents.extend(state)  # Never merge signed/unsigned Parts.
                    native_calls = [p["functionCall"] for content in state for p in content.get("parts", []) if "functionCall" in p]
                    for call, raw in zip(message.get("tool_calls") or [], native_calls):
                        call_names[call["id"]] = (raw["name"], raw.get("id"))
                else:
                    parts = [{"text": message_text(message)}] if message.get("content") else []
                    for call in message.get("tool_calls") or []:
                        function = call["function"]
                        parts.append({"functionCall": {"id": call["id"], "name": function["name"], "args": json.loads(function["arguments"])}})
                        call_names[call["id"]] = (function["name"], call["id"])
                    contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": "user", "parts": [{"text": message_text(message)}]})
        request = {"systemInstruction": {"parts": [{"text": "\n".join(system)}]}, "contents": contents,
                   "generationConfig": {"temperature": self.config.temperature, "maxOutputTokens": self.config.max_output_tokens}}
        if tools:
            # parametersJsonSchema supports the JSON Schema already used by tools.
            request["tools"] = [{"functionDeclarations": [{"name": t["function"]["name"],
                "description": t["function"].get("description", ""), "parametersJsonSchema": t["function"]["parameters"]} for t in tools]}]
            request["toolConfig"] = {"functionCallingConfig": {"mode": "ANY" if tool_choice == "required" else "AUTO"}}
        return request

    def _from_contents(self, contents, usage, reason):
        text, calls = [], []
        for content in contents:
            for part in content.get("parts", []):
                if "functionCall" in part:
                    call = part["functionCall"]
                    if not call.get("id") and self.config.model.startswith("gemini-3"):
                        raise ModelError("invalid_response", "Gemini 3 工具调用缺少原始 ID")
                    calls.append(ToolCall(call.get("id") or f"gemini_call_{len(calls)}", call["name"],
                                          json.dumps(call.get("args", {}), ensure_ascii=False)))
                elif "text" in part and not part.get("thought"):
                    text.append(part["text"])
                elif "text" not in part:
                    raise ModelError("unsupported_response", "Gemini 返回未启用的多模态或服务端工具内容")
        if reason == "MAX_TOKENS" and calls:
            raise ModelError("incomplete_tool_call", "Gemini 工具请求被截断，未执行工具")
        return Completion("".join(text), validate_calls(calls), normalize_usage(usage, "promptTokenCount", "candidatesTokenCount", "totalTokenCount"),
                          "length" if reason == "MAX_TOKENS" else reason, native_state(self.config, contents))

    def _decode(self, data):
        if data.get("error") or not data.get("candidates"):
            raise ModelError("provider_error", "Gemini 未返回可用候选结果")
        candidate = data["candidates"][0]
        reason = candidate.get("finishReason")
        if reason not in ("STOP", "MAX_TOKENS"):
            raise ModelError("model_refusal", "Gemini 未完成可用回答或触发内容限制")
        return self._from_contents([candidate.get("content") or {"role": "model", "parts": []}], data.get("usageMetadata") or {}, reason)

    def _accumulate(self, events, token):
        contents, usage, reason = [], {}, None
        for data in events:
            if data.get("error"):
                raise ModelError("provider_error", "Gemini 流返回失败事件")
            usage.update(data.get("usageMetadata") or {})
            for candidate in (data.get("candidates") or [])[:1]:
                content = candidate.get("content")
                if content:
                    contents.append(deepcopy(content))
                    for part in content.get("parts", []):
                        if "text" in part and not part.get("thought"):
                            yield ModelEvent("text_delta", part["text"])
                reason = candidate.get("finishReason", reason)
        if token:
            token.check()
        if reason is None:
            raise ModelError("incomplete_stream", "Gemini 流中断，未提交部分结果")
        if reason not in ("STOP", "MAX_TOKENS"):
            raise ModelError("model_refusal", "Gemini 流未生成可用回答")
        yield ModelEvent("completed", completion=self._from_contents(contents, usage, reason))
