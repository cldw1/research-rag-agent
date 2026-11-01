"""Streamlit startup with missing local resources; no model or LLM needed."""

from pathlib import Path
import socket
from unittest.mock import Mock

import pytest


def test_missing_key_and_qasper_index_allow_switching_to_pdf(monkeypatch):
    # AppTest creates an asyncio loop using local socket pairs. Probe with
    # bounded blocking sockets first: some Windows sandboxes otherwise hang
    # inside socketpair() before AppTest's own timeout starts.
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

    from streamlit.testing.v1 import AppTest
    from src.retrieval import validation_paper_scoped_retriever as validation
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(validation, "ValidationPaperScopedRetriever", Mock(side_effect=FileNotFoundError("index absent")))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert any("QASPER" in message.value for message in app.error)
    app.sidebar.radio[0].set_value("上传自己的 PDF").run()
    assert not app.exception
    assert not app.error
    assert any("上传 PDF" in message.value for message in app.info)
