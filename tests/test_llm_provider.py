import socket
from types import SimpleNamespace
from unittest.mock import Mock

import httpx2 as httpx
import pytest
from openai import APIStatusError

from src.llm.provider import ModelConfig, ModelError, OpenAICompatibleProvider, check_public_target


def public_resolver(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]


@pytest.mark.parametrize("url", ["http://example.com", "https://user:password@example.com", "https://example.com?key=credential", "https://example.com#fragment"])
def test_sensitive_or_non_https_urls_are_not_valid_config(url):
    with pytest.raises(ModelError):
        ModelConfig(provider="openai_compatible", base_url=url)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1"])
def test_private_and_metadata_targets_are_rejected(address):
    resolver = lambda *args, **kwargs: [(socket.AF_INET6 if ":" in address else socket.AF_INET, 1, 6, "", (address, 443))]
    with pytest.raises(ModelError, match="拒绝本机、内网"):
        check_public_target("https://example.com", resolver)


def test_deepseek_extra_parameters_do_not_leak_to_other_compatible_services():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer", tool_calls=[]), finish_reason="stop")], usage=None)
    for name in ("deepseek", "openai_compatible"):
        client = Mock()
        client.chat.completions.create.return_value = response
        config = ModelConfig(provider=name)
        provider = OpenAICompatibleProvider(config, "do-not-persist-this-credential", client=client, resolver=public_resolver)
        assert provider.complete([{"role": "user", "content": "q"}]).content == "answer"
        request = client.chat.completions.create.call_args.kwargs
        assert ("extra_body" in request) == (name == "deepseek")
        assert "api_key" not in config.to_dict()
        assert "credential" not in str(request)


def test_authentication_error_does_not_expose_provider_response_or_key():
    credential = "do-not-persist-this-credential"
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.com", headers={"Authorization": credential}))
    client = Mock()
    client.chat.completions.create.side_effect = APIStatusError(credential, response=response, body={"key": credential})
    provider = OpenAICompatibleProvider(ModelConfig(), credential, client=client, resolver=public_resolver)
    with pytest.raises(ModelError) as error:
        provider.complete([{"role": "user", "content": "q"}])
    assert error.value.code == "authentication_failed" and credential not in str(error.value)


def test_future_config_subclass_does_not_serialize_credential_fields():
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class CredentialConfig(ModelConfig):
        api_key: str = "do-not-persist-this-credential"
    assert "api_key" not in CredentialConfig().to_dict()


def test_actual_installed_sdk_client_accepts_transport_configuration_without_requests():
    provider = OpenAICompatibleProvider(ModelConfig(), "dummy-offline-credential")
    try:
        assert provider.client._client.follow_redirects is False
    finally:
        provider.close()
