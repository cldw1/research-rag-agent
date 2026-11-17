"""Offline verification of branch-head selection, prompts and evidence metrics."""

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.check_query_comparison import check_comparison
from src.evaluation.pdf_answers import read_json
from src.evaluation.pdf_benchmark import sha256
from src.evaluation.multidoc_qa import resolve_cases,score_context
from src.retrieval.branch_heads import select_branch_heads
from src.generation.strict_context_builder import build_strict_context
from src.generation.context_builder import build_qa_prompt


def check_head_run(run):
    run=Path(run);report=read_json(run/'summary.json');protocol=read_json(run/'protocol.json')
    if report['protocol']!=protocol:raise ValueError('Protocol differs')
    for name,h in report['artifacts_sha256'].items():
        if Path(name).name!=name or sha256((run/name).read_bytes())!=h:raise ValueError('Artifact changed')
    for name,h in report['source_code_sha256'].items():
        p=(run/'source_snapshot'/name).resolve()
        if (run/'source_snapshot').resolve() not in p.parents or sha256(p.read_bytes())!=h:raise ValueError('Snapshot changed')
    query_run=ROOT/'data/pdf_benchmark/query_runs'/protocol['query_run_id'];old=check_comparison(query_run)
    if sha256((query_run/'comparison.json').read_bytes())!=report['comparison_sha256']:raise ValueError('Baseline changed')
    retrieval=ROOT/'data/pdf_benchmark/runs'/old['protocol']['retrieval_run']
    chunks=read_json(run/'chunks.json');original=read_json(retrieval/'chunks.json')
    if {c['chunk_id']:c for c in chunks}!={c['chunk_id']:c for rows in original.values() for c in rows}:raise ValueError('Corpus changed')
    cases={c['question_id']:c for c in resolve_cases(read_json(ROOT/'benchmarks/multidoc_qa/questions.json'),read_json(retrieval/'dataset.json'))}
    builds={b['document_id']:b['index_id'] for b in read_json(retrieval/'summary.json')['builds']}
    lookup={c['chunk_id']:{**c,'persistent_index_id':builds[c['paper_id']]} for c in chunks};alignments={}
    for a in read_json(retrieval/'alignments.json'):alignments.setdefault(a['question_id'],[]).append(a)
    traces=read_json(run/'candidate_traces.json');prompts=read_json(run/'answer_prompts.json')
    by_prompt={(p['question_id'],p['variant']):p for p in prompts}
    by_record={r['question_id']:r for r in report['records']}
    if len(traces)!=len(cases) or {t['question_id'] for t in traces}!=set(cases) or set(by_record)!=set(cases) or len(by_prompt)!=len(cases)*3:
        raise ValueError('Population changed')
    for trace in traces:
        qid=trace['question_id'];case=cases[qid];record=by_record[qid]
        if trace['bindings']!=record['bindings']:raise ValueError('Document binding changed')
        fused={did:[lookup[cid] for cid in trace['fused_ids'][did]] for did in trace['document_ids']}
        for key in protocol['variants']:
            if key=='rrf':
                selected=[]
                for rank in range(5):
                    for did in trace['document_ids']:
                        if rank<len(fused[did]) and len(selected)<5:selected.append(fused[did][rank])
            else:selected=select_branch_heads(trace['document_ids'],fused,trace['branch_heads'],first_branch='bm25' if key=='bm25_first' else 'dense')
            context,sources=build_strict_context(selected,max_chars=7400);variant=record['variants'][key]
            if sources!=variant['sources'] or [c['chunk_id'] for c in selected]!=variant['selected_chunk_ids'] or len(context)!=variant['context_chars']:raise ValueError('Selection changed')
            if score_context(sources,chunks,case['targets'],trace['bindings'],alignments)!=variant['metrics']:raise ValueError('Metrics differ')
            if by_prompt[(qid,key)]!={'question_id':qid,'variant':key,'prompt':build_qa_prompt(case['question'],context),'sources':sources}:raise ValueError('Prompt differs')
            if key=='rrf' and sources!=next(r for r in old['records'] if r['question_id']==qid)['variants']['balanced_original']['sources']:raise ValueError('Control differs')
    for key in protocol['variants']:
        for field,stats in report['variants'][key].items():
            values=[r['variants'][key]['metrics'][field] for r in report['records'] if r['variants'][key]['metrics'][field] is not None]
            if stats!={'n':len(values),'mean':sum(values)/len(values) if values else None}:raise ValueError('Aggregate differs')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    result=check_head_run(parser.parse_args().run)
    print({'verified_questions':result['question_count'],'generation_api_calls':0})
