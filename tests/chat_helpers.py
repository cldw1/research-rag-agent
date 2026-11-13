from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from src.ingestion.structured_chunker import ByteBudgetCounter, ChunkConfig
from src.llm.provider import Completion, ModelConfig, ToolCall
from src.storage.conversations import ConversationStore
from src.storage.document_store import LocalDocumentStore
from src.storage.embedding_profile import EmbeddingProfile


class FixtureEmbedding:
    max_seq_length = 512
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text) + 2)))

    def encode(self, texts, **kwargs):
        vectors = np.array([[1., float("momentum" in t.lower()), float("experiment" in t.lower())]
                            for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeProvider:
    """Test-only responses, no network or claim about actual model quality."""

    def __init__(self, config, responses):
        self.config, self.responses, self.requests = config, list(responses), []

    def complete(self, messages, **kwargs):
        self.requests.append({"messages": deepcopy(messages), **deepcopy(kwargs)})
        if not self.responses:
            raise AssertionError("Unexpected model request")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_resources(root):
    store = LocalDocumentStore(root)
    document = store.ingest(b"# Methods\n\nMomentum uses a position update.\n\n# Results\n\nExperiment measures accuracy.", "paper.md").document
    model, counter = FixtureEmbedding(), ByteBudgetCounter()
    profile = EmbeddingProfile("fixture", "v1", 3, counter.identity, 512)
    store.build_index(document.document_id, model, profile, counter, ChunkConfig(strategy="structure"))
    retriever = store.load_retriever(document.document_id, model, profile)
    return SimpleNamespace(store=store, conversations=ConversationStore(store), model=model,
        counter=counter, profile=profile, document=document, retriever=retriever,
        config=ModelConfig(max_output_tokens=256))


def tool_answer(resources, *, call_id="call1", answer="Momentum uses a position update [1]."):
    import json
    return FakeProvider(resources.config, [
        Completion(tool_calls=(ToolCall(call_id, "search_research_paper", json.dumps({
            "paper_id": resources.document.document_id, "question": "momentum position"})),)),
        Completion(answer, usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}),
    ])
