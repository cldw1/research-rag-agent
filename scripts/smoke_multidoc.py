"""Two-process multi-document integration with cached BGE and scripted generation."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def worker(root, phase):
    from unittest.mock import Mock
    import torch
    from src.chat_service import ResearchChatService
    from src.embedding import load_embedding_resources
    from src.ingestion.parsers import UnifiedDocumentParser
    from src.llm.provider import Completion, ModelConfig, ToolCall
    from src.storage.backup import create_backup, restore_backup
    from src.storage.conversations import ConversationStore
    from src.storage.document_store import LocalDocumentStore

    torch.set_num_threads(8)
    model, counter, profile = load_embedding_resources(local_files_only=True)
    store = LocalDocumentStore(root)
    try:
        conversations = ConversationStore(store)
        state_path = root.parent / "state.json"
        if phase == "build":
            docs = []
            for filename, text in [
                ("momentum.md", b"# Methods\n\nThe study measures momentum with a position update."),
                ("control.md", b"# Methods\n\nThe experiment compares a control group on a fixed validation dataset.")]:
                doc = store.ingest(text, filename).document
                store.build_index(doc.document_id, model, profile, counter)
                docs.append(doc.document_id)
            cid = conversations.create(docs, ModelConfig(max_output_tokens=256), "Two synthetic papers")
            state = {"documents": docs, "conversation_id": cid}
            with state_path.open("x", encoding="utf-8") as stream:
                json.dump(state, stream)
        else:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            UnifiedDocumentParser.parse = Mock(side_effect=AssertionError("Unexpected reparse"))
            assert len(conversations.history(state["conversation_id"])) == 1
        encode = model.encode
        model.encode = Mock(side_effect=AssertionError("Restoring must not encode documents"))
        retriever = store.load_scoped_retriever(state["documents"], model, profile)
        model.encode.assert_not_called()
        spy = Mock(wraps=encode)
        model.encode = spy

        class ScriptedProvider:
            config = ModelConfig(max_output_tokens=256)
            requests = 0

            def complete(self, messages, **kwargs):
                self.requests += 1
                if self.requests == 1:
                    return Completion(tool_calls=(ToolCall("search_" + phase, "search_research_paper",
                        json.dumps({"paper_id": retriever.paper_id, "question": "Compare momentum and control group methods"})),))
                if self.requests != 2:
                    raise AssertionError("Unexpected generation request")
                payload = json.loads(messages[-1]["content"])
                assert set(payload["covered_document_ids"]) == set(state["documents"])
                assert [d["filename"] for d in payload["selected_documents"]] == ["momentum.md", "control.md"]
                assert "position update" in payload["evidence"] and "control group" in payload["evidence"]
                cites = []
                for did in state["documents"]:
                    source = next(s for s in payload["sources"] if s["paper_id"] == did
                        and "paragraph" in s["source_metadata"]["block_kinds"])
                    cites.append(source["citation"])
                return Completion(f"Synthetic paper A measures momentum [{cites[0]}]; paper B uses a control group [{cites[1]}].")

        provider = ScriptedProvider()
        result = ResearchChatService(conversations).ask(state["conversation_id"], "Compare the selected papers.",
            provider, retriever, phase)
        assert spy.call_count == 1 and provider.requests == 2
        assert set(result["evidence_document_ids"]) == set(state["documents"]) and not result["missing_document_ids"]
        sources = conversations.sources(state["conversation_id"])
        report = {"phase": phase, "pid": os.getpid(), "conversation_id": state["conversation_id"],
            "bindings": retriever.document_bindings, "selected_documents": len(state["documents"]),
            "document_encoding_calls_on_scope_load": 0, "real_bge_query_encoding_calls": spy.call_count,
            "history_turns": len(conversations.history(state["conversation_id"])),
            "cited_sources": [{key: s[key] for key in ("citation", "paper_id", "chunk_id", "persistent_index_id")}
                              for s in sources if s["citation"] in result["cited_numbers"]]}
        if phase == "restore":
            model.encode = Mock(side_effect=AssertionError("Backup restore must not encode"))
            backup_root, restored_root = root.parent / "backup", root.parent / "restored"
            create_backup(store, backup_root)
            restore_backup(backup_root, restored_root)
            restored = LocalDocumentStore(restored_root)
            try:
                copy = ConversationStore(restored)
                assert copy.history(state["conversation_id"]) == conversations.history(state["conversation_id"])
                assert copy.sources(state["conversation_id"]) == sources
                restored.load_scoped_retriever(state["documents"], model, profile)
                model.encode.assert_not_called()
                report["backup_restores_both_documents_and_history"] = True
            finally:
                restored.close()
        with (root.parent / (phase + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["build", "restore"])
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.root, args.worker)
        return
    parent = ROOT / "data/multidoc_smoke"
    parent.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    root = folder / "catalog"
    for phase in ("build", "restore"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--root", str(root)],
                       cwd=ROOT, check=True)
    build, restored = [json.loads((folder / (phase + ".json")).read_text(encoding="utf-8")) for phase in ("build", "restore")]
    assert build["pid"] != restored["pid"]
    assert build["bindings"] == restored["bindings"] and build["cited_sources"] == restored["cited_sources"]
    assert (build["history_turns"], restored["history_turns"]) == (1, 2)
    report = {"dataset": "two fixed synthetic Markdown papers", "embedding": "cached real BGE-small-en-v1.5, CPU",
        "llm_api_calls": 0, "generation": "scripted responses; no answer-quality claim",
        "processes": [build, restored], "stable_sources_after_restart": True,
        "scope": "Lifecycle, source coverage and version isolation integration, not a multi-paper retrieval benchmark."}
    with (folder / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("MULTIDOC_REPORT=" + str(folder / "summary.json"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
