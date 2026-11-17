"""Offline, same-budget comparison of RRF and branch-leader priority."""

import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import os
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.check_query_comparison import check_comparison
from src.evaluation.pdf_answers import read_json,write_new
from src.evaluation.pdf_benchmark import sha256
from src.evaluation.multidoc_qa import resolve_cases,score_context
from src.retrieval.branch_heads import select_branch_heads


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Output exists')
    protocol_path=ROOT/'benchmarks/multidoc_qa/head_protocol.json';protocol=read_json(protocol_path)
    query_run=ROOT/'data/pdf_benchmark/query_runs'/protocol['query_run_id']
    old=check_comparison(query_run);retrieval=ROOT/'data/pdf_benchmark/runs'/old['protocol']['retrieval_run']
    cases=resolve_cases(read_json(ROOT/'benchmarks/multidoc_qa/questions.json'),read_json(retrieval/'dataset.json'))
    if len(cases)!=protocol['question_count'] or (protocol['top_k'],protocol['max_chars'],protocol['candidate_k_per_branch_per_document'],protocol['rrf_k'])!=(5,7400,50,60):
        raise ValueError('Protocol differs from executable configuration')
    run=ROOT/'data/pdf_benchmark/head_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True);shutil.copytree(retrieval/'catalog',run/'catalog')
    (run/'protocol.json').write_bytes(protocol_path.read_bytes())
    names=['scripts/compare_branch_heads.py','src/retrieval/branch_heads.py','src/retrieval/multi_document_retriever.py',
        'src/retrieval/structured_retriever.py','src/evaluation/multidoc_qa.py','src/evaluation/pdf_benchmark.py',
        'src/generation/context_builder.py','src/generation/strict_context_builder.py','src/embedding.py']
    source_hashes={}
    for name in names:
        content=(ROOT/name).read_bytes();source_hashes[name]=sha256(content)
        p=run/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from src.embedding import load_embedding_resources
    from src.storage.document_store import LocalDocumentStore
    from src.retrieval.multi_document_retriever import MultiDocumentRetriever
    from src.generation.strict_context_builder import build_strict_context
    from src.generation.context_builder import build_qa_prompt
    torch.set_num_threads(8);model,_,profile=load_embedding_resources(local_files_only=True)
    base=read_json(retrieval/'summary.json')
    if profile.to_dict()!=base['embedding_profile']:raise ValueError('Embedding changed')
    alignments={}
    for a in read_json(retrieval/'alignments.json'):alignments.setdefault(a['question_id'],[]).append(a)
    store=LocalDocumentStore(run/'catalog');members={};records=[];traces=[];prompts=[];chunks=[]
    try:
        frozen_chunks=read_json(retrieval/'chunks.json')
        for b in base['builds']:
            m=store.load_retriever(b['document_id'],model,profile)
            if m.index_id!=b['index_id'] or m.chunks!=frozen_chunks[b['paper_id']]:raise ValueError('Saved index changed')
            members[b['paper_id']]=m;chunks.extend(m.chunks)
        for case in cases:
            pids=[t['paper_id'] for t in case['targets']];retriever=MultiDocumentRetriever([members[p] for p in pids])
            bindings={p:members[p].paper_id for p in pids};did_order=retriever.document_ids
            full=retriever.search(case['question'],retriever.paper_id,top_k=len(retriever.chunks),candidate_k=50)
            fused={did:sorted([c for c in full if c['paper_id']==did],key=lambda c:c['global_fusion_rank']) for did in did_order}
            heads={did:{} for did in did_order}
            for branch in ('bm25','dense'):
                hits=retriever.search(case['question'],retriever.paper_id,top_k=len(retriever.chunks),candidate_k=50,method=branch)
                for did in did_order:
                    owned=[c for c in hits if c['paper_id']==did]
                    heads[did][branch]=min(owned,key=lambda c:c[branch+'_rank'])['chunk_id'] if owned else None
            trace={'question_id':case['question_id'],'paper_ids':pids,'document_ids':list(did_order),'bindings':bindings,
                'fused_ids':{did:[c['chunk_id'] for c in fused[did]] for did in did_order},'branch_heads':heads}
            traces.append(trace);variants={}
            for variant in protocol['variants']:
                selected=full[:5] if variant=='rrf' else select_branch_heads(did_order,fused,heads,first_branch='bm25' if variant=='bm25_first' else 'dense')
                context,sources=build_strict_context(selected,max_chars=7400)
                if variant=='rrf':
                    before=next(r for r in old['records'] if r['question_id']==case['question_id'])['variants']['balanced_original']
                    if sources!=before['sources']:raise ValueError('Baseline did not reproduce')
                leaders={cid for d in heads.values() for cid in d.values() if cid is not None}
                variants[variant]={'selected_chunk_ids':[c['chunk_id'] for c in selected],'sources':sources,
                    'context_chars':len(context),'unique_branch_heads':len(leaders),
                    'heads_selected':len(leaders&{c['chunk_id'] for c in selected}),
                    'heads_in_context':len(leaders&{s['chunk_id'] for s in sources}),
                    'metrics':score_context(sources,retriever.chunks,case['targets'],bindings,alignments)}
                prompts.append({'question_id':case['question_id'],'variant':variant,'prompt':build_qa_prompt(case['question'],context),'sources':sources})
            records.append({'question_id':case['question_id'],'bindings':bindings,'variants':variants})
    finally:store.close()
    aggregate={}
    for key in protocol['variants']:
        aggregate[key]={}
        for field in ['document_presence','mean_full_evidence_recall','all_required_papers_hit','all_gold_complete']:
            values=[r['variants'][key]['metrics'][field] for r in records if r['variants'][key]['metrics'][field] is not None]
            aggregate[key][field]={'n':len(values),'mean':sum(values)/len(values) if values else None}
    for name,h in source_hashes.items():
        if sha256((ROOT/name).read_bytes())!=h:raise ValueError('Code changed during run')
    write_new(run/'chunks.json',chunks);write_new(run/'candidate_traces.json',traces);write_new(run/'answer_prompts.json',prompts)
    baseline_rows={r['question_id']:r for r in records}
    changes={}
    for variant in protocol['variants'][1:]:
        pairs=[(r['variants']['rrf']['metrics']['mean_full_evidence_recall'],r['variants'][variant]['metrics']['mean_full_evidence_recall']) for r in records if r['variants']['rrf']['metrics']['mean_full_evidence_recall'] is not None]
        changes[variant]={'improved':sum(b>a for a,b in pairs),'unchanged':sum(b==a for a,b in pairs),'worsened':sum(b<a for a,b in pairs)}
    report={'scope':protocol['scope'],'run_id':run.name,'protocol':protocol,'question_count':len(cases),
        'generation_api_calls':0,'answer_quality':None,'human_review':'pending','variants':aggregate,'changes':changes,'records':records,
        'source_code_sha256':source_hashes,'comparison_sha256':sha256((query_run/'comparison.json').read_bytes()),
        'artifacts_sha256':{n:sha256((run/n).read_bytes()) for n in ['protocol.json','chunks.json','candidate_traces.json','answer_prompts.json']},
        'note':'At most five anchors. Both predeclared branch orders retained. No gold-conditioned selection or per-question winner. No production default change.'}
    write_new(run/'summary.json',report);write_new(args.output,report)
    print({'run':str(run),'variants':aggregate,'changes':changes})


if __name__=='__main__':main()
