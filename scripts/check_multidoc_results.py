"""Replay saved multi-document prompts and metrics without models or API calls."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pdf_answers import load_inputs, read_json
from src.evaluation.pdf_benchmark import sha256
from src.evaluation.multidoc_qa import resolve_cases, score_context
from src.generation.context_builder import build_qa_prompt
from src.generation.strict_context_builder import build_strict_context


def check_run(run, retrieval):
    run, retrieval = Path(run), Path(retrieval)
    report = read_json(run / "summary.json")
    for name, expected in report["artifacts_sha256"].items():
        if Path(name).name != name or sha256((run/name).read_bytes()) != expected:
            raise ValueError("Comparison artifact changed")
    for name, expected in report["source_code_sha256"].items():
        path = (run / "source_snapshot" / name).resolve()
        if (run / "source_snapshot").resolve() not in path.parents or sha256(path.read_bytes()) != expected:
            raise ValueError("Comparison source snapshot changed")
    dataset, _ = load_inputs(retrieval)
    if (run / "dataset.json").read_bytes() != (retrieval / "dataset.json").read_bytes():
        raise ValueError("Different baseline dataset")
    if read_json(retrieval / "summary.json")["embedding_profile"] != report["embedding_profile"]:
        raise ValueError("Different embedding model")
    cases = {c["question_id"]: c for c in resolve_cases(read_json(run/"questions.json"), dataset)}
    chunks = read_json(retrieval / "chunks.json")
    builds = {b["paper_id"]:b for b in read_json(retrieval/"summary.json")["builds"]}
    alignments = {}
    for a in read_json(retrieval/"alignments.json"):
        alignments.setdefault(a["question_id"], []).append(a)
    requests = read_json(run/"generation_requests.json")
    reqs = {(r["question_id"],r["variant"]):r for r in requests}
    if len(reqs) != len(requests) or len(reqs) != len(cases)*len(report["protocol"]["variants"]):
        raise ValueError("Generation-input population changed")
    records = read_json(run/"records.json")
    if len(records) != len(cases) or {r["question_id"] for r in records} != set(cases):
        raise ValueError("Question population changed")
    for record in records:
        case = cases[record["question_id"]]
        bindings = {t["paper_id"]:builds[t["paper_id"]]["document_id"] for t in case["targets"]}
        if bindings != record["bindings"]:
            raise ValueError("Comparison document bindings changed")
        canonical = [{**c,"persistent_index_id":builds[p]["index_id"]} for p in bindings for c in chunks[p]]
        lookup = {c["chunk_id"]:c for c in canonical}
        for key, variant in record["variants"].items():
            context, sources = build_strict_context([lookup[s["chunk_id"]] for s in variant["sources"]], max_chars=7400)
            prompt = build_qa_prompt(case["question"],context)
            req = reqs[(record["question_id"],key)]
            if (sources != variant["sources"] or req["sources"] != sources or req["prompt"] != prompt
                    or req["prompt_sha256"] != sha256(prompt.encode()) or variant["context_chars"] != len(context)):
                raise ValueError("Prompt or source scope changed")
            scores = score_context(sources,canonical,case["targets"],bindings,alignments)
            if scores != variant["metrics"]:
                raise ValueError("Per-paper metrics differ")
    aggregate = {}
    for key in report["protocol"]["variants"]:
        aggregate[key] = {}
        for field in ("document_presence","mean_full_evidence_recall","all_required_papers_hit","all_gold_complete"):
            values = [r["variants"][key]["metrics"][field] for r in records if r["variants"][key]["metrics"][field] is not None]
            aggregate[key][field] = {"n":len(values),"mean":sum(values)/len(values) if values else None}
    if aggregate != report["variants"] or records != report["records"]:
        raise ValueError("Summary differs from saved records")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--retrieval-run",type=Path,default=ROOT/"data/pdf_benchmark/runs/20261115T165227Z-1bd0349b")
    args=parser.parse_args()
    report=check_run(args.run,args.retrieval_run)
    print({"verified_questions":report["question_count"],"verified_prompts":report["question_count"]*len(report["variants"]),"generation_api_calls":0})


if __name__ == "__main__":
    main()
