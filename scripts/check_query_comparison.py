"""Replay query-comparison inputs and evidence metrics, without model requests."""

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.evaluation.pdf_answers import read_json,collect_results,summarize_answers
from src.evaluation.pdf_benchmark import sha256
from src.evaluation.multidoc_qa import resolve_cases,score_context
from src.retrieval.document_queries import parse_document_queries
from src.generation.strict_context_builder import build_strict_context
from src.generation.context_builder import build_qa_prompt
from scripts.check_multidoc_results import check_run


def check_comparison(run):
    run=Path(run);manifest=read_json(run/'manifest.json');report=read_json(run/'comparison.json')
    for name,h in manifest['inputs_sha256'].items():
        if sha256((ROOT/name).read_bytes())!=h: raise ValueError('Input changed')
    for name,h in manifest['source_code_sha256'].items():
        path=(run/'source_snapshot'/name).resolve()
        if (run/'source_snapshot').resolve() not in path.parents or sha256(path.read_bytes())!=h: raise ValueError('Snapshot changed')
    for name,h in report['artifacts_sha256'].items():
        if Path(name).name!=name or sha256((run/name).read_bytes())!=h: raise ValueError('Artifact changed')
    for qid,h in report['planner_result_sha256'].items():
        if sha256((run/'attempts'/qid/'result.json').read_bytes())!=h: raise ValueError('Planner response changed')
    requests=read_json(run/'requests.json');results=collect_results(run,requests)
    if summarize_answers(results,manifest)!=report['planner_generation']: raise ValueError('Token/request accounting changed')
    protocol=report['protocol'];retrieval=ROOT/'data/pdf_benchmark/runs'/protocol['retrieval_run']
    baseline=check_run(ROOT/'data/pdf_benchmark/multidoc_runs'/protocol['baseline_run'],retrieval)
    cases={c['question_id']:c for c in resolve_cases(read_json(ROOT/'benchmarks/multidoc_qa/questions.json'),read_json(retrieval/'dataset.json'))}
    chunks=read_json(retrieval/'chunks.json');builds={b['paper_id']:b for b in read_json(retrieval/'summary.json')['builds']}
    alignments={}
    for a in read_json(retrieval/'alignments.json'): alignments.setdefault(a['question_id'],[]).append(a)
    responses={r['question_id']:r for r in results};prompts={(p['question_id'],p['variant']):p for p in read_json(run/'answer_prompts.json')}
    records=read_json(run/'retrieval_records.json')
    if len(records)!=len(cases) or {r['question_id'] for r in records}!=set(cases): raise ValueError('Population changed')
    for record in records:
        qid=record['question_id'];case=cases[qid]
        bindings={t['paper_id']:builds[t['paper_id']]['document_id'] for t in case['targets']}
        if record['bindings']!=bindings: raise ValueError('Scope changed')
        if record['planner_status']=='valid' and parse_document_queries(responses[qid]['answer'],bindings.values())!=record['queries']:
            raise ValueError('Subqueries differ from actual model output')
        if record['planner_status']!='valid' and record['queries']!={did:case['question'] for did in bindings.values()}:
            raise ValueError('Fallback query changed')
        canonical=[{**c,'persistent_index_id':builds[p]['index_id']} for p in bindings for c in chunks[p]]
        lookup={c['chunk_id']:c for c in canonical}
        for key,variant in record['variants'].items():
            context,sources=build_strict_context([lookup[s['chunk_id']] for s in variant['sources']],max_chars=7400)
            if sources!=variant['sources'] or len(context)!=variant['context_chars']: raise ValueError('Context changed')
            if variant['metrics']!=score_context(sources,canonical,case['targets'],bindings,alignments): raise ValueError('Metrics changed')
            if prompts[(qid,key)]!={'question_id':qid,'variant':key,'prompt':build_qa_prompt(case['question'],context),'sources':sources}:
                raise ValueError('Answer input changed')
        old=next(r for r in baseline['records'] if r['question_id']==qid)['variants']['balanced_top5']
        if old['sources']!=record['variants']['balanced_original']['sources']: raise ValueError('Baseline changed')
    if records!=report['records']: raise ValueError('Published records differ')
    for key in protocol['variants']:
        for field,stats in report['variants'][key].items():
            values=[r['variants'][key]['metrics'][field] for r in records if r['variants'][key]['metrics'][field] is not None]
            if stats!={'n':len(values),'mean':sum(values)/len(values) if values else None}: raise ValueError('Aggregate changed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    report=check_comparison(parser.parse_args().run)
    print({'verified_questions':report['question_count'],'planner_tokens':report['planner_generation']['known_returned_usage']['total_tokens'],'model_requests_made_by_checker':0})
