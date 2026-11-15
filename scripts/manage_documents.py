"""Persistent document management. No generation API or destructive migration."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("jobs")
    commands.add_parser("recover")
    commands.add_parser("schema-version")
    commands.add_parser("upgrade-chat")
    commands.add_parser("upgrade-multidoc")
    upload = commands.add_parser("upload")
    upload.add_argument("file", type=Path)
    upload.add_argument("--ocr", action="store_true")
    upload.add_argument("--tessdata")
    upload.add_argument("--language", default="eng")
    upload.add_argument("--backend", choices=["lightweight", "docling"], default="lightweight")
    upload.add_argument("--artifacts")
    upload.add_argument("--formula", action="store_true")
    for name in ("reindex", "query", "delete", "undelete"):
        command = commands.add_parser(name)
        command.add_argument("document_id")
        if name in ("reindex", "query"):
            command.add_argument("--local-files-only", action="store_true")
        if name == "reindex":
            command.add_argument("--force", action="store_true")
            command.add_argument("--strategy", choices=["adaptive", "hybrid", "structure", "recursive", "fixed"], default="adaptive")
        if name == "query":
            command.add_argument("question")
            command.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
    backup = commands.add_parser("backup")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("source", type=Path)
    restore.add_argument("destination", type=Path)
    args = parser.parse_args()
    from src.storage.document_store import LocalDocumentStore
    from src.storage.artifacts import StorageError

    store = None
    try:
        if args.command == "restore":
            from src.storage.backup import restore_backup
            result = {"restored": str(restore_backup(args.source, args.destination))}
        else:
            store = LocalDocumentStore(args.data_dir)
            if args.command == "list":
                result = store.list_documents(include_deleted=True)
            elif args.command == "jobs":
                result = store.list_jobs()
            elif args.command == "recover":
                result = {"interrupted_jobs": store.recover_expired_jobs()}
            elif args.command == "schema-version":
                from src.storage.database import catalog_version
                result = {"schema_version": catalog_version(store.engine)}
            elif args.command == "upgrade-chat":
                from src.storage.database import catalog_version, upgrade_chat_schema
                backup = upgrade_chat_schema(store)
                result = {"schema_version": catalog_version(store.engine), "backup": str(backup) if backup else None}
            elif args.command == "upgrade-multidoc":
                from src.storage.database import catalog_version, upgrade_multidoc_schema
                backup = upgrade_multidoc_schema(store)
                result = {"schema_version": catalog_version(store.engine), "backup": str(backup) if backup else None}
            elif args.command == "upload":
                from src.ingestion.parsers import ParseOptions
                options = ParseOptions(ocr=args.ocr, ocr_language=args.language, tessdata=args.tessdata,
                    backend=args.backend, docling_artifacts_path=args.artifacts, docling_formula=args.formula)
                if args.file.stat().st_size > options.max_bytes:
                    raise StorageError("file_too_large", "文件超过大小上限")
                record = store.ingest(args.file.read_bytes(), args.file.name, options)
                result = {"document_id": record.document.document_id, "status": record.document.status, "reused": record.reused}
            elif args.command in ("delete", "undelete"):
                action = store.delete_document if args.command == "delete" else store.restore_document
                action(args.document_id)
                result = {"status": "ok"}
            elif args.command == "backup":
                from src.storage.backup import create_backup
                manifest = create_backup(store, args.destination)
                result = {"files": len(manifest["files"]), "backup": str(args.destination)}
            else:
                from src.embedding import load_embedding_resources
                model, counter, profile = load_embedding_resources(local_files_only=args.local_files_only)
                if args.command == "reindex":
                    from dataclasses import asdict
                    from src.ingestion.structured_chunker import ChunkConfig
                    result = asdict(store.build_index(args.document_id, model, profile, counter,
                        ChunkConfig(strategy=args.strategy), force=args.force))
                else:
                    retriever = store.load_retriever(args.document_id, model, profile)
                    result = retriever.search(args.question, args.document_id, method=args.method)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        # Do not echo arbitrary filenames, provider errors or full local paths.
        print(f"{getattr(exc, 'code', 'operation_failed')}: 文档操作未完成，请检查参数、文件或索引状态。", file=sys.stderr)
        return 1
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
