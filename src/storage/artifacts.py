"""Checksummed, immutable index directories; no pickle or uploaded FAISS input."""

import hashlib
import json
import os
import uuid
from pathlib import Path


class StorageError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise StorageError("unsafe_path", "存储路径无效")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise StorageError("unsafe_path", "存储路径超出数据目录")
    return target


def save_object(root, digest, content):
    relative = f"objects/{digest}"
    path = safe_path(root, relative)
    if path.exists():
        if file_hash(path) != digest:
            raise StorageError("object_corrupt", "已保存原文件校验失败，未覆盖原文件")
        return relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)  # Concurrent writers have the same content hash.
    finally:
        if temporary.exists():
            temporary.unlink()  # Only this operation's generated temporary file.
    return relative


def save_index(root, index_id, retriever, profile):
    import numpy as np
    from src.runtime import load_faiss

    faiss = load_faiss()
    relative = f"indexes/{index_id}"
    directory = safe_path(root, relative)
    directory.mkdir(parents=True, exist_ok=False)
    # Python I/O avoids native FAISS path handling issues on Unicode Windows paths.
    (directory / "index.faiss").write_bytes(faiss.serialize_index(retriever.index).tobytes())
    with (directory / "vectors.npy").open("xb") as stream:
        np.save(stream, retriever.vectors, allow_pickle=False)
    manifest = {
        "format_version": 1, "index_id": index_id,
        "document_id": retriever.document.document_id,
        "document_version": retriever.document.document_version,
        "chunk_count": len(retriever.chunks), "dimension": retriever.vectors.shape[1],
        "chunks_sha256": fingerprint(retriever.chunks), "embedding_profile": profile,
        "files": {name: {"sha256": file_hash(directory / name), "bytes": (directory / name).stat().st_size}
                  for name in ("index.faiss", "vectors.npy")},
    }
    (directory / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
    return relative, manifest


def verify_index(root, relative, expected):
    directory = safe_path(root, relative)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest != expected or set(expected["files"]) != {"index.faiss", "vectors.npy"}:
            raise ValueError("manifest mismatch")
        for name, info in expected["files"].items():
            path = safe_path(root, f"{relative}/{name}")
            if path.stat().st_size != info["bytes"] or file_hash(path) != info["sha256"]:
                raise ValueError("artifact mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StorageError("index_corrupt", "索引文件缺失或校验失败，请重新构建；未读取损坏的 FAISS 数据") from exc
    return directory
