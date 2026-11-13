"""Fresh-process check: no manually configured Anaconda path in the command."""

import subprocess
import sys
from pathlib import Path


def test_faiss_loads_in_fresh_project_process():
    process = subprocess.run([sys.executable, "-c",
        "from src.runtime import load_faiss; f=load_faiss(); assert f.IndexFlatIP(3).d == 3; print('READY')"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert "READY" in process.stdout
