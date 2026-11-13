"""Model identity includes the pipeline/tokenizer, beyond matching dimensions."""

import json
from types import SimpleNamespace

import torch

from src.ingestion.structured_chunker import ModelTokenCounter
from src.storage.embedding_profile import EmbeddingProfile


def test_tokenizer_normalization_changes_identity_but_transient_padding_does_not():
    configuration = {"normalizer": {"lowercase": True}, "padding": None}
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 1}, special_tokens_map={"unk_token": "[UNK]"},
        backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(configuration)))
    initial = ModelTokenCounter(tokenizer, "test").identity
    configuration["padding"] = {"strategy": "BatchLongest"}
    assert ModelTokenCounter(tokenizer, "test").identity == initial
    configuration["normalizer"]["lowercase"] = False
    assert ModelTokenCounter(tokenizer, "test").identity != initial


def test_embedding_weights_and_pooling_config_both_change_profile():
    settings = {"pooling": "mean"}
    tensor = torch.tensor([[1., 2.]])
    model = SimpleNamespace(state_dict=lambda: {"w": tensor}, max_seq_length=512,
        get_embedding_dimension=lambda: 2,
        _modules={"pooling": SimpleNamespace(get_config_dict=lambda: dict(settings))})
    counter = SimpleNamespace(identity="test-tokenizer")
    initial = EmbeddingProfile.from_model(model, counter, "test")
    settings["pooling"] = "cls"
    assert EmbeddingProfile.from_model(model, counter, "test") != initial
    settings["pooling"] = "mean"
    tensor[0, 0] = 3
    assert EmbeddingProfile.from_model(model, counter, "test") != initial
