from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.test_workbench_startup import require_app_test_sockets


@pytest.mark.parametrize("existing_chat", [False, True])
def test_chat_history_is_browsable_without_key_or_embedding(monkeypatch, tmp_path, existing_chat):
    require_app_test_sockets()
    from streamlit.testing.v1 import AppTest
    from src import embedding
    from src.chat_service import ResearchChatService
    from src.storage.document_store import LocalDocumentStore
    from tests.chat_helpers import make_resources, tool_answer

    root = tmp_path / "catalog"
    if existing_chat:
        item = make_resources(root)
        cid = item.conversations.create(item.document.document_id, item.config)
        ResearchChatService(item.conversations).ask(cid, "question", tool_answer(item), item.retriever, "first")
        item.store.close()
    else:
        documents = LocalDocumentStore(root)
        documents.close()
    monkeypatch.setenv("COPILOT_DATA_DIR", str(root))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    loader = Mock(side_effect=AssertionError("History browsing must not load BGE"))
    monkeypatch.setattr(embedding, "load_embedding_resources", loader)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / "2_research_chat.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    loader.assert_not_called()
    if existing_chat:
        assert len(app.chat_message) >= 2
        regenerate = next(button for button in app.button if button.label == "重新生成最近回答")
        retry = next(button for button in app.button if button.label == "重试最近未完成问题")
        assert regenerate.disabled and retry.disabled  # Viewing without a key never starts a request.
    else:
        assert any("新建会话" in message.value for message in app.info)


def test_model_selector_switches_all_providers_without_loading_embedding(monkeypatch, tmp_path):
    require_app_test_sockets()
    from streamlit.testing.v1 import AppTest
    from src import embedding
    from src.llm.factory import CATALOG
    from src.storage.document_store import LocalDocumentStore

    root = tmp_path / "catalog"
    documents = LocalDocumentStore(root)
    documents.close()
    monkeypatch.setenv("COPILOT_DATA_DIR", str(root))
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.setenv(name, "")
    loader = Mock(side_effect=AssertionError("Selecting models must not load BGE or call a model"))
    monkeypatch.setattr(embedding, "load_embedding_resources", loader)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / "2_research_chat.py"), default_timeout=30).run()
    for name in CATALOG:
        selector = next(item for item in app.selectbox if item.label == "供应商")
        selector.select(name).run()
        assert not app.exception and not app.error
        assert next(item for item in app.text_input if item.label == "API Key").value == ""
        assert next(item for item in app.checkbox if item.label == "流式输出").value
    loader.assert_not_called()
