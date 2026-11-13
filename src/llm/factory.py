"""Credential-free catalog and provider construction."""

from src.llm.provider import OpenAICompatibleProvider


CATALOG = {
    "DeepSeek": {"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-flash", "key_env": "DEEPSEEK_API_KEY"},
    "OpenAI Responses": {"provider": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-5.4", "key_env": "OPENAI_API_KEY"},
    "Anthropic Claude": {"provider": "anthropic", "base_url": "https://api.anthropic.com", "model": "claude-opus-5-5", "key_env": "ANTHROPIC_API_KEY"},
    "Google Gemini": {"provider": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta", "model": "gemini-3.8-flash", "key_env": "GEMINI_API_KEY"},
    "阿里云百炼": {"provider": "bailian", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "key_env": "DASHSCOPE_API_KEY"},
    "自定义 OpenAI 兼容接口": {"provider": "openai_compatible", "base_url": "https://", "model": "", "key_env": None},
}


def create_provider(config, api_key, **kwargs):
    from src.llm.native import AnthropicProvider, GeminiProvider, OpenAIResponsesProvider
    implementation = {"openai": OpenAIResponsesProvider, "anthropic": AnthropicProvider,
                      "gemini": GeminiProvider}.get(config.provider, OpenAICompatibleProvider)
    return implementation(config, api_key, **kwargs)
