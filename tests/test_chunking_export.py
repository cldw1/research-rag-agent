import hashlib
import json

import pytest

from scripts.export_chunking_comparison import export_comparison
from src.evaluation.chunking_comparison import summarize_comparison
from tests.test_chunking_comparison import population


def completed_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    records = population()
    protocol = {"split": "validation", "evidence_char_budget": 7400,
        "configs": {s: {"max_tokens": 480, "max_chars": 1200} for s in ("adaptive", "structure")},
        "embedding_profile": {"dimension": 2}}
    build = dict(chunk_count=2, chunk_tokens=[4, 5], chunk_characters=[10, 20],
                 vector_bytes=16, merged_paragraph_chunks=0, cache_hit=True)
    papers = [{"paper_id": f"p{i}", "strategies": {s: dict(build) for s in ("adaptive", "structure")}} for i in range(2)]
    engineering = {s: {key: 2 * build[key] for key in ("chunk_count", "vector_bytes", "merged_paragraph_chunks", "cache_hit")}
                   for s in ("adaptive", "structure")}
    contents = {"protocol.json": json.dumps(protocol), "per_paper.json": json.dumps(papers), "excluded.json": "[]",
                "per_question.jsonl": "\n".join(json.dumps(r) for r in records) + "\n"}
    hashes = {}
    for name, value in contents.items():
        raw = value.encode("utf-8")
        (run / name).write_bytes(raw)
        hashes[{"protocol.json": "protocol_sha256", "per_paper.json": "per_paper_sha256",
            "excluded.json": "excluded_sha256", "per_question.jsonl": "per_question_sha256"}[name]] = hashlib.sha256(raw).hexdigest()
    report = {"protocol": protocol, "artifacts": hashes, "question_count": len(records),
        "corpus": {"eligible_questions": len(records), "eligible_papers": 2, "total_papers": 2}, "engineering": engineering,
        "adaptive_baseline_reproduced": True, "all_source_characters_preserved": True,
        "production_default_unchanged": True, **summarize_comparison(records)}
    (run / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    return run


def test_numeric_export_hashes_rows_and_refuses_overwriting(tmp_path):
    run = completed_run(tmp_path)
    destination = tmp_path / "public"
    result = export_comparison(run, destination)
    assert result["questions"] == 4
    report = json.loads((destination / "qasper_chunking_validation.json").read_text(encoding="utf-8"))
    for name, metadata in report["public_artifacts"].items():
        raw = (destination / name).read_bytes()
        assert b"\r\n" not in raw
        assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
        assert len(raw.splitlines()) - 1 == metadata["rows"]
    with pytest.raises(FileExistsError):
        export_comparison(run, destination)


def test_export_rejects_tampered_raw_records(tmp_path):
    run = completed_run(tmp_path)
    with (run / "per_question.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        export_comparison(run, tmp_path / "public")


def test_export_rejects_unreproducible_aggregate(tmp_path):
    run = completed_run(tmp_path)
    path = run / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["results"]["structure"]["methods"]["hybrid"]["5"]["hit"] = .99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="do not reproduce"):
        export_comparison(run, tmp_path / "public")
