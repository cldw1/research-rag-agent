"""Small normalized model contract and DeepSeek/OpenAI-compatible adapter."""

import ipaddress
import re
import socket
from dataclasses import dataclass, field, fields
from typing import Protocol
from urllib.parse import urlsplit


class ModelError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def check_url(url):
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or any(c.isspace() for c in url)):
            raise ValueError("unsafe URL")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("invalid port")
    except ValueError:
        raise ModelError("invalid_base_url", "Base URL 必须是无认证信息、查询参数或片段的 HTTPS 地址") from None
    return parsed


def check_public_target(url, resolver=None):
    parsed = check_url(url)
    try:
        results = (resolver or socket.getaddrinfo)(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        addresses = [ipaddress.ip_address(item[4][0]) for item in results]
    except (OSError, ValueError):
        raise ModelError("target_unavailable", "无法校验模型服务地址，请检查域名和网络") from None
    if not addresses or any(not address.is_global for address in addresses):
        raise ModelError("private_target", "当前模式只允许公网模型服务，拒绝本机、内网及元数据地址")


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "deepseek"
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_output_tokens: int = 1500
    context_window: int = 32768  # Application budget, not a claim about the model maximum.
    safety_margin: int = 512
    supports_tools: bool = True
    stream: bool = False

    def __post_init__(self):
        if self.provider not in ("deepseek", "openai_compatible", "openai", "anthropic", "gemini", "bailian"):
            raise ModelError("unsupported_provider", "不支持该模型供应商")
        if not re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}", self.model):
            raise ModelError("invalid_model", "模型名称无效")
        check_url(self.base_url)
        if self.provider == "deepseek" and self.base_url.rstrip("/") != "https://api.deepseek.com":
            raise ModelError("invalid_base_url", "DeepSeek 模式使用官方地址；其他地址请选择自定义接口")
        official = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com",
                    "gemini": "https://generativelanguage.googleapis.com/v1beta"}
        if self.provider in official and self.base_url.rstrip("/") != official[self.provider]:
            raise ModelError("invalid_base_url", "该原生供应商模式使用官方地址")
        if self.provider == "bailian":
            parsed = urlsplit(self.base_url)
            host = parsed.hostname or ""
            if not (host == "dashscope.aliyuncs.com" or re.fullmatch(r"[a-zA-Z0-9-]+\.(?:cn-beijing|ap-southeast-1|us-east-1)\.maas\.aliyuncs\.com", host)) or parsed.path.rstrip("/") != "/compatible-mode/v1":
                raise ModelError("invalid_base_url", "百炼地址应使用匹配地域/工作空间的官方 compatible-mode/v1 地址")
        if (not 0 <= self.temperature <= 2 or not 16 <= self.max_output_tokens <= 32768
                or not 512 <= self.context_window <= 1048576 or self.safety_margin < 128
                or self.max_output_tokens + self.safety_margin >= self.context_window):
            raise ModelError("invalid_budget", "模型参数或上下文/输出预算无效")

    def to_dict(self):
        # Explicitly use BASE configuration fields, even if a future subclass
        # adds credential fields. Never serialize a client or arbitrary __dict__.
        return {item.name: getattr(self, item.name) for item in fields(ModelConfig)}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_message(self):
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass(frozen=True)
class Completion:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    provider_state: dict = field(default_factory=dict)
    response_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvent:
    kind: str  # text_delta / completed
    text: str = ""
    completion: Completion | None = None


def message_text(message):
    text = message.get("content") or ""
    if message.get("reference_context"):
        text += "\n\nUNTRUSTED RETRIEVED REFERENCE DATA:\n" + message["reference_context"]
    return text


def wire_messages(messages):
    output = []
    for message in messages:
        record = {key: message[key] for key in ("role", "content", "tool_calls", "tool_call_id") if key in message}
        if message.get("reference_context"):
            record["content"] = message_text(message)
        output.append(record)
    return output


def validate_calls(calls):
    if (len(calls) > 16 or len({c.id for c in calls}) != len(calls)
            or any(not c.id or len(c.id) > 256 or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", c.name)
                   or not isinstance(c.arguments, str) or len(c.arguments) > 16000 for c in calls)):
        raise ModelError("invalid_response", "模型返回的工具调用结构无效")
    return tuple(calls)


class LLMProvider(Protocol):
    config: ModelConfig
    def complete(self, messages, *, tools=None, tool_choice=None) -> Completion: ...


class OpenAICompatibleProvider:
    supports_required_tools = True
    def __init__(self, config, api_key, *, client=None, resolver=None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ModelError("missing_key", "请填写当前模型服务的 API Key")
        self.config, self.resolver = config, resolver
        self._owns_client = client is None
        self._http_client = None
        if client is None:
            from openai import DefaultHttpxClient, OpenAI
            self._http_client = DefaultHttpxClient(follow_redirects=False, trust_env=False)
            client = OpenAI(api_key=api_key.strip(), base_url=config.base_url,
                timeout=60.0, max_retries=1, http_client=self._http_client)
        self.client = client

    def close(self):
        if self._owns_client:
            self.client.close()

    def complete(self, messages, *, tools=None, tool_choice=None):
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        check_public_target(self.config.base_url, self.resolver)
        request = {"model": self.config.model, "messages": wire_messages(messages),
                   "temperature": self.config.temperature, "max_tokens": self.config.max_output_tokens,
                   "stream": False}
        if self.config.provider == "deepseek":
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        elif self.config.provider == "bailian":
            request["extra_body"] = {"enable_thinking": False}
        if tools:
            if not self.config.supports_tools:
                raise ModelError("tools_unsupported", "当前配置不支持 Tool Calling，请使用普通 RAG")
            request.update(tools=tools, tool_choice=tool_choice or "auto")
        try:
            response = self.client.chat.completions.create(**request)
            message = response.choices[0].message
            calls = tuple(ToolCall(c.id, c.function.name, c.function.arguments) for c in (message.tool_calls or []))
            validate_calls(calls)
            if response.choices[0].finish_reason == "length" and calls:
                raise ModelError("incomplete_tool_call", "工具调用输出被截断，未执行工具")
            usage = {}
            if response.usage:
                for name in ("prompt_tokens", "completion_tokens", "total_tokens",
                             "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                    value = getattr(response.usage, name, None)
                    if isinstance(value, int) and value >= 0:
                        usage[name] = value
            metadata = {name: value for name in ("id", "model", "system_fingerprint")
                        if isinstance(value := getattr(response, name, None), str)}
            return Completion(message.content or "", calls, usage, response.choices[0].finish_reason,
                              response_metadata=metadata)
        except APITimeoutError:
            raise ModelError("model_timeout", "模型请求超时，请稍后重试") from None
        except APIConnectionError:
            raise ModelError("model_connection", "无法连接模型服务，请检查地址和网络") from None
        except APIStatusError as exc:
            code = "authentication_failed" if exc.status_code in (401, 403) else "rate_limited" if exc.status_code == 429 else "provider_error"
            raise ModelError(code, "模型服务拒绝请求，请检查授权、额度或模型参数") from None
        except ModelError:
            raise
        except Exception:
            raise ModelError("invalid_response", "模型返回了无法处理的结果") from None

    def stream_complete(self, messages, *, tools=None, tool_choice=None, cancellation=None):
        from src.llm.streaming import stream_chat_completions
        yield from stream_chat_completions(self, messages, tools, tool_choice, cancellation)
