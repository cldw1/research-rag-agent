"""Independent-process chat recovery: real cached BGE, explicitly mocked LLM."""

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


class ScriptedProvider:
    """Integration fixture only. Never contacts or evaluates a real LLM."""

    def __init__(self, config, document_id):
        self.config, self.document_id, self.calls = config, document_id, 0

    def complete(self, messages, **kwargs):
        from src.llm.provider import Completion, ToolCall
        self.calls += 1
        if self.calls == 1:
            return Completion(tool_calls=(ToolCall("search_call", "search_research_paper", json.dumps({
                "paper_id": self.document_id, "question": "fixed validation dataset recall"})),))
        return Completion("Synthetic test answer referring to the retrieved evidence [1].")


def worker(root, phase):
    from src.chat_service import ResearchChatService
    from src.embedding import load_embedding_resources
    from src.llm.provider import ModelConfig
    from src.storage.backup import create_backup, restore_backup
    from src.storage.conversations import ConversationStore
    from src.storage.document_store import LocalDocumentStore

    documents = LocalDocumentStore(root)
    chats = ConversationStore(documents)
    config = ModelConfig(max_output_tokens=256)
    model, counter, profile = load_embedding_resources(local_files_only=True)
    if phase == "write":
        document = documents.ingest(b"# Method\n\nThe fixed validation dataset is used to measure recall.\n\n# Limitations\n\nThis tiny synthetic example cannot establish answer accuracy.", "chat_fixture.md").document
        documents.build_index(document.document_id, model, profile, counter)
        cid = chats.create(document.document_id, config, "Synthetic chat")
    else:
        cid = chats.list()[0]["id"]
        document = documents.get_document(chats.details(cid)["document_id"])
        assert len(chats.history(cid)) == 1
    retriever = documents.load_retriever(document.document_id, model, profile)
    provider = ScriptedProvider(config, document.document_id)
    result = ResearchChatService(chats).ask(cid, "What is the validation dataset used for?", provider, retriever, phase)
    output = {"pid": os.getpid(), "conversation_id": cid, "phase": phase,
              "turns": len(chats.history(cid)), "messages": sum(len(turn["messages"]) for turn in chats.history(cid)),
              "citation_numbers": result["cited_numbers"], "sources": len(chats.sources(cid)),
              "generation": "scripted_mock_not_real_model", "mock_model_calls": provider.calls,
              "invalid_citations": result["invalid_citations"], "context": result["context_selections"]}
    if phase == "read":
        assert len(chats.history(cid)) == 2 and result["cited_numbers"] == [1]
        assert "Synthetic test answer" in chats.export_markdown(cid)
        backup, restored = root.parent / "backup", root.parent / "restored"
        create_backup(documents, backup)
        restore_backup(backup, restored)
        restored_documents = LocalDocumentStore(restored)
        try:
            recovered = ConversationStore(restored_documents)
            assert recovered.history(cid) == chats.history(cid)
            assert recovered.sources(cid) == chats.sources(cid)
            output["backup_restored_chat_and_citations"] = True
        finally:
            restored_documents.close()
    documents.close()
    (root.parent / f"{phase}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["write", "read"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v3_chat_smoke.json")
    args = parser.parse_args()
    if args.worker:
        worker(args.root, args.worker)
        return 0
    parent = ROOT / "data" / "copilot_v3_smoke"
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    for phase in ("write", "read"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--root", str(workspace / "catalog")],
            cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, check=True)
    records = [json.loads((workspace / f"{phase}.json").read_text(encoding="utf-8")) for phase in ("write", "read")]
    assert records[0]["pid"] != records[1]["pid"] and records[0]["conversation_id"] == records[1]["conversation_id"]
    report = {"dataset": "one fixed synthetic Markdown, two scripted questions", "paid_llm_calls": 0,
              "embedding": "real local BGE-small-en-v1.5, CPU", "records": records,
              "limitations": "Persistence/control-flow integration only. Generation was mocked; no real answer quality or live provider test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
