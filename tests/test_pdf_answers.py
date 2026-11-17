"""Billable request idempotency, poisoned prompt checks and honest accounting."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.evaluation.pdf_answers import (
    attempt_paths, citation_diagnostics, collect_results, generate_one, load_inputs,
    summarize_answers, write_new,
)
from src.evaluation.pdf_benchmark import sha256
from src.llm.provider import Completion, ModelConfig, ModelError, OpenAICompatibleProvider
from tests.test_llm_provider import public_resolver


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_RUN = ROOT / "data/pdf_benchmark/runs/20261115T165227Z-1bd0349b"


def request():
    return {"question_id": "q1", "paper_id": "p1", "prompt_sha256": "fixed",
            "messages": [{"role": "user", "content": "Evidence only"}], "input_estimate": 100,
            "sources": [{"citation": 1, "paper_id": "p1", "chunk_id": "c1"}]}


def provider(response=None):
    return SimpleNamespace(config=ModelConfig(), complete=Mock(return_value=response or Completion(
        content="Supported answer [1].", usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        finish_reason="stop")))


def test_success_is_durable_and_resume_does_not_bill_twice(tmp_path):
    p = provider()
    first = generate_one(tmp_path, request(), p)
    second = generate_one(tmp_path, request(), p)
    assert first == second and first["status"] == "succeeded"
    assert first["citation_format_valid"] and p.complete.call_count == 1
    summary = summarize_answers(collect_results(tmp_path, [request()]), {"model": "fixture"})
    assert summary["generation_api_attempts"] == 1 and summary["known_returned_usage"]["total_tokens"] == 28
    assert summary["answer_quality_metrics"] is None


def test_uncertain_call_is_not_automatically_replayed(tmp_path):
    intent, _ = attempt_paths(tmp_path, "q1")
    write_new(intent, {"question_id": "q1", "prompt_sha256": "fixed"})
    p = provider()
    result = generate_one(tmp_path, request(), p)
    assert result["status"] == "uncertain"
    p.complete.assert_not_called()


def test_error_records_only_stable_code_and_does_not_leak_credentials(tmp_path):
    p = provider()
    p.complete.side_effect = RuntimeError("private credential and request payload")
    result = generate_one(tmp_path, request(), p)
    assert result["status"] == "failed" and result["error_code"] == "generation_failed"
    assert "private credential" not in json.dumps(result)
    # Failed calls are not implicitly retried on resume either.
    generate_one(tmp_path, request(), p)
    assert p.complete.call_count == 1


def test_input_budget_is_checked_before_creating_intent_or_request(tmp_path):
    r = request()
    r["input_estimate"] = 1000000
    p = provider()
    with pytest.raises(ValueError, match="budget"):
        generate_one(tmp_path, r, p)
    p.complete.assert_not_called()
    assert not attempt_paths(tmp_path, "q1")[0].exists()


@pytest.mark.parametrize("completion,status", [(Completion("", finish_reason="stop"), "empty"),
    (Completion("Unfinished [1]", finish_reason="length"), "truncated")])
def test_empty_and_truncated_responses_are_not_successes(tmp_path, completion, status):
    assert generate_one(tmp_path, request(), provider(completion))["status"] == status


def test_reference_numbers_are_only_format_diagnostics():
    sources = [{"citation": 1}, {"citation": 2}]
    result = citation_diagnostics("Claim [1, 2], invented ref [9].", sources)
    assert result["cited_numbers"] == [1, 2, 9] and result["invalid_citations"] == [9]
    assert "faithfulness" not in result and "correctness" not in result


def test_unknown_usage_is_not_fabricated_as_a_cost(tmp_path):
    result = generate_one(tmp_path, request(), provider(Completion("Answer [1]", finish_reason="stop")))
    summary = summarize_answers([result], {})
    assert summary["estimated_returned_cost_usd_upper_at_published_peak_rates"] is None
    assert not summary["usage_complete_for_returned_responses"]


def test_provider_captures_observed_model_and_cache_usage_without_arbitrary_metadata():
    response = SimpleNamespace(id="completion-id", model="observed-model", system_fingerprint="fingerprint",
        credential="must-not-leak", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13,
            prompt_cache_hit_tokens=7, prompt_cache_miss_tokens=3),
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer", tool_calls=[]), finish_reason="stop")])
    client = Mock()
    client.chat.completions.create.return_value = response
    p = OpenAICompatibleProvider(ModelConfig(), "dummy-secret", client=client, resolver=public_resolver)
    result = p.complete([{"role": "user", "content": "q"}])
    assert result.response_metadata == {"id": "completion-id", "model": "observed-model", "system_fingerprint": "fingerprint"}
    assert result.usage["prompt_cache_hit_tokens"] == 7
    assert "must-not-leak" not in json.dumps(result.response_metadata)


@pytest.mark.skipif(not RETRIEVAL_RUN.exists(), reason="Optional local retrieval artifacts")
def test_real_frozen_inputs_cover_dev_only_and_never_include_answer_keys():
    dataset, requests = load_inputs(RETRIEVAL_RUN)
    assert len(requests) == 45 and {r["paper_id"] for r in requests} == {"attention", "rag", "lora"}
    assert all(set(r["messages"][0]) == {"role", "content"} for r in requests)
    assert all("reference_points" not in r and "answerable" not in r for r in requests)


def test_prompt_tampering_and_test_labels_are_rejected_before_generation(tmp_path):
    # A tiny self-contained run is enough to exercise guards before reconstruction.
    payload = {"schema_version": "pdf-qa-v1", "status": "annotation_candidate", "papers": [
        {"paper_id": "test", "split": "test", "filename": "t.pdf", "pdf_sha256": "0"*64, "page_count": 1}],
        "questions": []}
    write_new(tmp_path / "dataset.json", payload)
    write_new(tmp_path / "summary.json", {"artifacts_sha256": {"dataset.json": sha256((tmp_path / "dataset.json").read_bytes())}})
    with pytest.raises(ValueError, match="development"):
        load_inputs(tmp_path)
    write_new(tmp_path / "changed.txt", "original")
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    summary["artifacts_sha256"]["changed.txt"] = "0"*64
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load_inputs(tmp_path)
