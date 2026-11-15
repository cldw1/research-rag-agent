from pathlib import Path
from unittest.mock import Mock

from tests.test_multidoc import collection, provider
from tests.test_workbench_startup import require_app_test_sockets


def test_multiple_documents_can_be_selected_and_saved_without_model_loading(collection, monkeypatch):
    require_app_test_sockets()
    from streamlit.testing.v1 import AppTest
    from src import embedding
    r = collection
    monkeypatch.setenv("COPILOT_DATA_DIR", str(r.store.root))
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.setenv(name, "")
    loader = Mock(side_effect=AssertionError("Selecting documents must not encode or call a model"))
    monkeypatch.setattr(embedding, "load_embedding_resources", loader)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages/2_research_chat.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    app.multiselect[0].set_value(r.ids).run()
    next(button for button in app.button if button.label == "新建会话").click().run()
    assert not app.exception and not app.error
    cid = app.session_state["research_conversation_id"]
    assert cid != r.cid
    assert [d["document_id"] for d in r.conversations.details(cid)["documents"]] == r.ids
    assert any("paper.md" in item.value and "second.md" in item.value for item in app.caption)
    loader.assert_not_called()


def test_multi_document_history_displays_distinct_source_filenames_without_key(collection, monkeypatch):
    require_app_test_sockets()
    from streamlit.testing.v1 import AppTest
    from src import embedding
    from src.chat_service import ResearchChatService
    r = collection
    ResearchChatService(r.conversations).ask(r.cid, "compare", provider(r), r.retriever, "first")
    monkeypatch.setenv("COPILOT_DATA_DIR", str(r.store.root))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(embedding, "load_embedding_resources", Mock(side_effect=AssertionError("History must not load BGE")))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages/2_research_chat.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    labels = [item.label for item in app.expander]
    assert any("[1]" in value and "paper.md" in value for value in labels)
    assert any("[2]" in value and "second.md" in value for value in labels)


def test_existing_v2_library_can_enable_multiple_selection_after_explicit_backup(tmp_path, monkeypatch):
    require_app_test_sockets()
    from streamlit.testing.v1 import AppTest
    from src import embedding
    from src.storage.database import catalog_version
    from tests.test_multidoc_migration import v2_library
    r = v2_library(tmp_path / "old")
    try:
        second = r.store.ingest(b"# Methods\n\nA second indexed experiment.", "second.md").document
        r.store.build_index(second.document_id, r.model, r.profile, r.counter)
        monkeypatch.setenv("COPILOT_DATA_DIR", str(r.store.root))
        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        loader = Mock(side_effect=AssertionError("Schema upgrade must not load BGE"))
        monkeypatch.setattr(embedding, "load_embedding_resources", loader)
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages/2_research_chat.py"), default_timeout=30).run()
        assert not app.exception and catalog_version(r.store.engine) == 2
        next(button for button in app.button if button.label == "备份并启用多文档会话").click().run()
        assert not app.exception and not app.error and catalog_version(r.store.engine) == 3
        app.multiselect[0].set_value([r.document.document_id, second.document_id]).run()
        next(button for button in app.button if button.label == "新建会话").click().run()
        cid = app.session_state["research_conversation_id"]
        assert len(r.conversations.details(cid)["documents"]) == 2
        assert list((r.store.root.parent / "catalog_backups").glob("*/backup_manifest.json"))
        loader.assert_not_called()
    finally:
        r.store.close()
