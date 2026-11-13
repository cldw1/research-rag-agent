"""Chat Completions stream accumulation. Partial tool JSON is never executed."""

from contextlib import contextmanager

from src.llm.provider import (Completion, ModelError, ModelEvent, ToolCall,
                              check_public_target, validate_calls, wire_messages)


@contextmanager
def close_on_cancel(resource, token):
    key = token.register(resource.close) if token else None
    try:
        if token:
            token.check()
        yield resource
    finally:
        if token:
            token.unregister(key)
        resource.close()


def stream_chat_completions(provider, messages, tools, tool_choice, token):
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    if token:
        token.check()
    check_public_target(provider.config.base_url, provider.resolver)
    request = {"model": provider.config.model, "messages": wire_messages(messages),
        "temperature": provider.config.temperature, "max_tokens": provider.config.max_output_tokens,
        "stream": True, "stream_options": {"include_usage": True}}
    if provider.config.provider == "deepseek":
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    elif provider.config.provider == "bailian":
        request["extra_body"] = {"enable_thinking": False}
    if tools:
        if not provider.config.supports_tools:
            raise ModelError("tools_unsupported", "当前服务配置不支持工具")
        request.update(tools=tools, tool_choice=tool_choice or "auto")
    text, calls, usage, finished, reason, size = [], {}, {}, False, None, 0
    try:
        stream = provider.client.chat.completions.create(**request)
        with close_on_cancel(stream, token):
            for chunk in stream:
                if token:
                    token.check()
                if getattr(chunk, "usage", None):
                    usage = {name: value for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                             if isinstance((value := getattr(chunk.usage, name, None)), int) and value >= 0}
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    reason, finished = choice.finish_reason, True
                delta = choice.delta
                if delta.content:
                    text.append(delta.content)
                    size += len(delta.content)
                    yield ModelEvent("text_delta", delta.content)
                for call in delta.tool_calls or []:
                    record = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                    if call.id:
                        if record["id"] and record["id"] != call.id:
                            raise ModelError("invalid_response", "工具调用 ID 在流中发生变化")
                        record["id"] = call.id
                    if call.function:
                        record["name"] += call.function.name or ""
                        record["arguments"] += call.function.arguments or ""
                    if len(record["arguments"]) > 16000 or len(calls) > 16:
                        raise ModelError("invalid_response", "工具调用超过大小限制")
                if size > 2 * 1024 * 1024:
                    raise ModelError("response_too_large", "模型输出超过应用限制")
        if token:
            token.check()
        if not finished:
            raise ModelError("incomplete_stream", "模型流意外中断，未提交部分回答")
        if reason == "length" and calls:
            raise ModelError("incomplete_tool_call", "工具调用输出被截断，未执行工具")
        result = validate_calls([ToolCall(**calls[index]) for index in sorted(calls)])
        yield ModelEvent("completed", completion=Completion("".join(text), result, usage, reason))
    except Exception as exc:
        if token:
            token.check()
        if isinstance(exc, ModelError):
            raise
        if isinstance(exc, APITimeoutError):
            raise ModelError("model_timeout", "模型流读取超时") from None
        if isinstance(exc, APIConnectionError):
            raise ModelError("model_connection", "模型流连接失败") from None
        if isinstance(exc, APIStatusError):
            raise ModelError("provider_error", "模型服务拒绝流式请求，请检查权限、额度和参数") from None
        raise ModelError("invalid_response", "模型流返回了无法处理的事件") from None
