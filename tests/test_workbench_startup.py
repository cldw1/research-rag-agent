"""UI startup without models or API calls; requires local sockets for AppTest."""

import socket
from pathlib import Path
from unittest.mock import Mock

import pytest


def require_app_test_sockets():
    try:
        with socket.socket() as listener:
            listener.settimeout(1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with socket.create_connection(listener.getsockname(), timeout=1):
                connection, _ = listener.accept()
                connection.close()
    except OSError:
        pytest.skip("Local loopback sockets unavailable; Streamlit AppTest requires them")


@pytest.mark.parametrize("existing_document", [False, True])
def test_workbench_lists_saved_documents_without_embedding(monkeypatch, tmp_path, existing_document):
    require_app_test_sockets()
    from streamlit.testing.v1 import AppTest
    from src import embedding
    from src.storage.document_store import LocalDocumentStore

    root = tmp_path / "catalog"
    store = LocalDocumentStore(root)
    if existing_document:
        document = store.ingest(b"Saved paper text for offline preview.", "saved.txt").document
    store.close()
    monkeypatch.setenv("COPILOT_DATA_DIR", str(root))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    loader = Mock(side_effect=AssertionError("Browsing documents must not load models"))
    monkeypatch.setattr(embedding, "load_embedding_resources", loader)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / "1_document_workbench.py"), default_timeout=30).run()
    assert not app.exception and not app.error
    if existing_document:
        document_picker = next(item for item in app.selectbox if item.label == "当前文档")
        # The selected value is the stable document ID; format_func renders the filename.
        assert any("saved.txt" in option for option in document_picker.options)
        assert document_picker.value == document.document_id
    else:
        assert any("暂无匹配文件" in message.value for message in app.info)
    loader.assert_not_called()
