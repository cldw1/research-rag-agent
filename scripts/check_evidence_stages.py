"""Recompute the gold-unit stage audit from immutable local trace IDs."""

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.pdf_answers import read_json
from src.evaluation.pdf_benchmark import sha256
from src.evaluation.evidence_stages import probe_evidence,summarize_probes
from scripts.check_query_comparison import check_comparison


def check_stage_report(run):
    run=Path(run);report=read_json(run/'summary.json')
    for name,h in report['artifacts_sha256'].items():
        if Path(name).name!=name or sha256((run/name).read_bytes())!=h:raise ValueError('Audit artifact changed')
    for name,h in report['source_code_sha256'].items():
        path=(run/'source_snapshot'/name).resolve()
        if (run/'source_snapshot').resolve() not in path.parents or sha256(path.read_bytes())!=h:raise ValueError('Audit snapshot changed')
    query_run=ROOT/'data/pdf_benchmark/query_runs'/report['query_run_id']
    comparison=check_comparison(query_run)
    if sha256((query_run/'comparison.json').read_bytes())!=report['comparison_sha256']:raise ValueError('Different query experiment')
    chunks=read_json(run/'chunks.json');traces=read_json(run/'traces.json')
    retrieval=ROOT/'data/pdf_benchmark/runs'/comparison['protocol']['retrieval_run']
    original_chunks=read_json(retrieval/'chunks.json')
    if {c['chunk_id']:c for c in chunks}!={c['chunk_id']:c for rows in original_chunks.values() for c in rows}:
        raise ValueError('Audit corpus differs from the frozen retrieval chunks')
    original_alignments={}
    for a in read_json(retrieval/'alignments.json'):original_alignments.setdefault(a['question_id'],[]).append(a)
    for trace in traces:
        if trace['aligned']!=original_alignments[trace['source_question_id']]:raise ValueError('Audit gold coordinates changed')
    records=[probe_evidence([c for c in chunks if c['paper_id']==t['document_id']],t) for t in traces]
    if records!=report['records'] or summarize_probes(records)!=report['variants']:raise ValueError('Audit metrics differ')
    expected={(r['question_id'],k,p) for r in comparison['records'] for k,v in r['variants'].items() for p in v['metrics']['per_paper_gold']}
    actual={(r['question_id'],r['variant'],r['paper_id']) for r in records}
    if len(actual)!=len(records) or actual!=expected:raise ValueError('Evidence audit population changed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    report=check_stage_report(parser.parse_args().run)
    print({'verified_paper_variant_records':len(report['records']),'generation_api_calls':0})
