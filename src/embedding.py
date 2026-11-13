"""One local embedding resource configuration, separate from generation models."""

from pathlib import Path

from src.ingestion.structured_chunker import ModelTokenCounter
from src.storage.embedding_profile import EmbeddingProfile


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embedding_resources(*, local_files_only=False):
    from sentence_transformers import SentenceTransformer

    root = Path(__file__).resolve().parents[1]
    model = SentenceTransformer(MODEL_NAME, device="cpu", cache_folder=str(root / "models" / "embedding_cache"),
                               local_files_only=local_files_only)
    counter = ModelTokenCounter(model.tokenizer, MODEL_NAME)
    profile = EmbeddingProfile.from_model(model, counter, MODEL_NAME)
    return model, counter, profile
