"""Coherent SQLite + referenced immutable files snapshot, restored to a NEW root."""

import json
import shutil
import sqlite3
from pathlib import Path

from sqlalchemy import select
from types import SimpleNamespace

from src.storage.artifacts import StorageError, canonical, file_hash, safe_path, verify_index
from src.storage.models import DocumentRecord, IndexVersion


def create_backup(store, destination):
    destination = Path(destination).resolve()
    # Never merge into/overwrite existing files, including an existing backup.
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    # Lock catalog publication while reading the SQLite snapshot and file list.
    # Long-running computations may continue, but cannot activate new pointers.
    with store.engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            # Use this Connection directly so a nested ORM Session cannot release
            # or alter the outer snapshot/publication lock on close.
            originals = connection.execute(select(DocumentRecord.__table__)).mappings().all()
            indexes = connection.execute(select(IndexVersion.__table__)).mappings().all()
            paths = set()
            for document in originals:
                store._check_object(SimpleNamespace(**document))
                paths.add(document["object_path"])
            for index in indexes:
                verify_index(store.root, index["artifact_path"], index["manifest"])
                paths.update(f"{index['artifact_path']}/{name}"
                             for name in ("manifest.json", "index.faiss", "vectors.npy"))
            for relative in sorted(paths):
                source = safe_path(store.root, relative)
                target = safe_path(destination, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                files[relative] = file_hash(target)
            with sqlite3.connect(store.database_path) as source:
                with sqlite3.connect(destination / "catalog.sqlite") as target:
                    source.backup(target)
            files["catalog.sqlite"] = file_hash(destination / "catalog.sqlite")
        finally:
            connection.rollback()
    # Manifest is written last: incomplete backups do not look complete.
    manifest = {"backup_format": 1, "files": files}
    (destination / "backup_manifest.json").write_text(canonical(manifest), encoding="utf-8")
    return manifest


def verify_backup(directory):
    directory = Path(directory).resolve()
    try:
        manifest = json.loads((directory / "backup_manifest.json").read_text(encoding="utf-8"))
        if manifest["backup_format"] != 1 or "catalog.sqlite" not in manifest["files"]:
            raise ValueError("format mismatch")
        for relative, digest in manifest["files"].items():
            if file_hash(safe_path(directory, relative)) != digest:
                raise ValueError("checksum mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StorageError("backup_corrupt", "备份不完整或校验失败，未恢复数据") from exc
    return manifest


def restore_backup(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = verify_backup(source)
    destination.mkdir(parents=True, exist_ok=False)
    for relative in manifest["files"]:
        target = safe_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(source, relative), target)
        if file_hash(target) != manifest["files"][relative]:
            raise StorageError("backup_corrupt", "恢复文件校验失败，原备份未改动")
    # Running operations in a restored snapshot belong to the previous process.
    # Expire their leases; the service's explicit recovery then marks them interrupted.
    with sqlite3.connect(destination / "catalog.sqlite") as connection:
        connection.execute("UPDATE ingestion_jobs SET lease_until='1970-01-01T00:00:00+00:00' WHERE status='running'")
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversation_turns'").fetchone():
            connection.execute("UPDATE conversation_turns SET lease_until='1970-01-01T00:00:00+00:00' WHERE status='running'")
    return destination
