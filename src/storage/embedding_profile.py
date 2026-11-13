"""Generation-independent embedding identity for safe index reuse."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    model_name: str
    revision: str
    dimension: int
    tokenizer: str
    max_sequence_length: int
    normalized: bool = True
    metric: str = "inner_product"

    def __post_init__(self):
        if not self.model_name or not self.revision or not self.tokenizer or self.dimension < 1:
            raise ValueError("Embedding 配置必须包含模型、版本指纹、维度和 Tokenizer")
        if self.metric != "inner_product" or not self.normalized or self.max_sequence_length < 1:
            raise ValueError("当前持久化模式使用归一化向量与 inner_product")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_model(cls, model, counter, model_name):
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        # Pooling and prompt defaults can change vectors without changing weights.
        modules = [{"name": name, "type": type(module).__qualname__,
                    "config": module.get_config_dict() if hasattr(module, "get_config_dict") else {}}
                   for name, module in model._modules.items()]
        digest.update(json.dumps({"modules": modules, "prompts": getattr(model, "prompts", {}),
                                  "default_prompt": getattr(model, "default_prompt_name", None)},
                                 sort_keys=True, default=str).encode())
        dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return cls(model_name, "weights-config-sha256:" + digest.hexdigest(),
                   int(dimension()), counter.identity,
                   int(model.max_seq_length))
