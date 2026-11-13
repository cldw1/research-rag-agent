"""Two-process lifecycle smoke test with cached real BGE; never calls an LLM."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONTENT = b"# Methods\n\nThe experiment uses a fixed validation dataset and reports recall.\n\n# Limitations\n\nThis synthetic example cannot establish retrieval quality."


def worker(root, phase):
    from unittest.mock import Mock
    from src.embedding import load_embedding_resources
    from src.ingestion.parsers import UnifiedDocumentParser
    from src.storage.document_store import LocalDocumentStore
    from src.storage.artifacts import StorageError
    from src.storage.backup import create_backup, restore_backup

    model, counter, profile = load_embedding_resources(local_files_only=True)
    store = LocalDocumentStore(root)
    result = {"pid": os.getpid(), "phase": phase}
    try:
        if phase == "build":
            document = store.ingest(CONTENT, "fixture.md").document
            t0 = time.perf_counter()
            built = store.build_index(document.document_id, model, profile, counter)
            result.update(document_id=document.document_id, index_id=built.index_id,
                          chunks=built.chunk_count, build_seconds=time.perf_counter() - t0,
                          embedding_profile=profile.to_dict())
        else:
            # Fail immediately if restore/dedup accidentally reparses or re-embeds.
            UnifiedDocumentParser.parse = Mock(side_effect=AssertionError("restore reparsed a document"))
            model.encode = Mock(side_effect=AssertionError("restore encoded a document"))
            duplicate = store.ingest(CONTENT, "renamed.md")
            document_id = duplicate.document.document_id
            t0 = time.perf_counter()
            retriever = store.load_retriever(document_id, model, profile)
            hits = retriever.search("fixed validation dataset", document_id, method="bm25")
            reused = store.build_index(document_id, model, profile, counter)
            result.update(document_id=document_id, index_id=reused.index_id,
                restored_seconds=time.perf_counter() - t0, duplicate_reused=duplicate.reused,
                index_reused=reused.reused, recovered_expected_evidence=any("fixed validation dataset" in h["text"] for h in hits))
            backup_root, restored_root = root.parent / "backup", root.parent / "restored"
            create_backup(store, backup_root)
            restore_backup(backup_root, restored_root)
            restored = LocalDocumentStore(restored_root)
            try:
                result["backup_restores_without_embedding"] = bool(restored.load_retriever(
                    document_id, model, profile).search("dataset", document_id, method="bm25"))
            finally:
                restored.close()
            store.delete_document(document_id)
            try:
                retriever.search("dataset", document_id, method="bm25")
            except StorageError:
                result["cached_retriever_blocked_after_delete"] = True
            else:
                raise AssertionError("deleted document was retrievable")
            result["original_retained"] = (root / "objects" / duplicate.document.file_hash).read_bytes() == CONTENT
            model.encode.assert_not_called()
            result["document_embedding_calls_on_restore"] = 0
        (root.parent / (phase + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v2_persistence_smoke.json")
    parser.add_argument("--worker", choices=["build", "restore"])
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.root, args.worker)
        return 0
    parent = ROOT / "data" / "copilot_v2_smoke"
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    root = workspace / "catalog"
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for phase in ("build", "restore"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--root", str(root)],
                       cwd=ROOT, env=environment, check=True)
    build, restored = [json.loads((workspace / f"{phase}.json").read_text(encoding="utf-8"))
                       for phase in ("build", "restore")]
    assert build["pid"] != restored["pid"] and build["index_id"] == restored["index_id"]
    assert restored["duplicate_reused"] and restored["index_reused"]
    assert restored["recovered_expected_evidence"] and restored["backup_restores_without_embedding"]
    result = {"dataset": "one deterministic synthetic Markdown document", "embedding": "cached real BGE-small-en-v1.5 on CPU",
              "llm_calls": 0, "processes": [build, restored], "workspace": str(workspace),
              "limitations": "Lifecycle integration, not retrieval-quality or production throughput evidence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
