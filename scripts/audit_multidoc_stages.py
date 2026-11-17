"""Replay fixed multi-document queries and locate candidate/rank/quota/budget loss."""

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
from src.evaluation.multidoc_qa import resolve_cases
from src.evaluation.evidence_stages import probe_evidence,summarize_probes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-run',type=Path,default=ROOT/'data/pdf_benchmark/query_runs/20261116T081839Z-8cc4be13')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Output exists')
    comparison=check_comparison(args.query_run)
    retrieval=ROOT/'data/pdf_benchmark/runs'/comparison['protocol']['retrieval_run']
    cases=resolve_cases(read_json(ROOT/'benchmarks/multidoc_qa/questions.json'),read_json(retrieval/'dataset.json'))
    old_records={r['question_id']:r for r in comparison['records']}
    alignments={}
    for a in read_json(retrieval/'alignments.json'):alignments.setdefault(a['question_id'],[]).append(a)
    run=ROOT/'data/pdf_benchmark/stage_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    shutil.copytree(retrieval/'catalog',run/'catalog')
    names=['scripts/audit_multidoc_stages.py','src/evaluation/evidence_stages.py','src/evaluation/pdf_benchmark.py',
        'src/retrieval/structured_retriever.py','src/retrieval/multi_document_retriever.py',
        'src/generation/context_builder.py','src/generation/strict_context_builder.py','src/embedding.py']
    source_hashes={}
    for name in names:
        content=(ROOT/name).read_bytes();source_hashes[name]=sha256(content)
        path=run/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from src.embedding import load_embedding_resources
    from src.storage.document_store import LocalDocumentStore
    from src.retrieval.multi_document_retriever import MultiDocumentRetriever
    from src.generation.strict_context_builder import build_strict_context
    torch.set_num_threads(8)
    model,_,profile=load_embedding_resources(local_files_only=True)
    base=read_json(retrieval/'summary.json')
    if profile.to_dict()!=base['embedding_profile']:raise ValueError('Embedding profile changed')
    store=LocalDocumentStore(run/'catalog');members={};traces=[];probes=[];saved_chunks=[]
    try:
        for b in base['builds']:
            member=store.load_retriever(b['document_id'],model,profile)
            if member.index_id!=b['index_id']:raise ValueError('Index changed')
            members[b['paper_id']]=member;saved_chunks.extend(member.chunks)
        for case in cases:
            pids=[t['paper_id'] for t in case['targets']]
            combined=MultiDocumentRetriever([members[p] for p in pids]);did_order=combined.document_ids
            positions={c['chunk_id']:i for i,c in enumerate(combined.chunks)}
            for variant in comparison['protocol']['variants']:
                branches={};fused_by_doc={}
                if variant=='balanced_original':
                    for method in ('bm25','dense'):
                        hits=combined.search(case['question'],combined.paper_id,top_k=len(combined.chunks),candidate_k=50,method=method)
                        branches[method]={did:sorted([{'chunk_id':c['chunk_id'],'rank':c[method+'_rank']} for c in hits if c['paper_id']==did],key=lambda r:r['rank']) for did in did_order}
                    full=combined.search(case['question'],combined.paper_id,top_k=len(combined.chunks),candidate_k=50)
                    selected=full[:5]
                    for did in did_order:
                        fused_by_doc[did]=sorted([c for c in full if c['paper_id']==did],key=lambda c:c['global_fusion_rank'])
                else:
                    for did in did_order:
                        member=combined.document_retrievers[did]
                        query=case['question'] if variant=='scoped_original' else old_records[case['question_id']]['queries'][did]
                        by_id={c['chunk_id']:c for c in member.chunks};scores={}
                        for method in ('bm25','dense'):
                            hits=member.search(query,did,top_k=50,candidate_k=50,method=method)
                            ranked=[{'chunk_id':c['chunk_id'],'rank':c[method+'_rank']} for c in hits]
                            branches.setdefault(method,{})[did]=ranked
                            for h in ranked:scores[h['chunk_id']]=scores.get(h['chunk_id'],0)+1/(60+h['rank'])
                        fused_by_doc[did]=[by_id[cid] for cid in sorted(scores,key=lambda cid:(-scores[cid],positions[cid]))]
                        actual=member.search(query,did,top_k=5,candidate_k=50)
                        if [c['chunk_id'] for c in actual]!=[c['chunk_id'] for c in fused_by_doc[did][:5]]:raise ValueError('Fusion replay differs')
                    selected=[]
                    for rank in range(5):
                        for did in did_order:
                            if rank<len(fused_by_doc[did]) and len(selected)<5:selected.append(fused_by_doc[did][rank])
                selected=[{**c,'persistent_index_id':combined.document_bindings[c['paper_id']]} for c in selected]
                _,sources=build_strict_context(selected,max_chars=7400)
                expected=old_records[case['question_id']]['variants'][variant]['sources']
                if sources!=expected:raise ValueError('Selected/context sources did not reproduce')
                for target in case['targets']:
                    if not target['answerable']:continue
                    pid=target['paper_id'];did=members[pid].paper_id;fused=fused_by_doc[did]
                    trace={'question_id':case['question_id'],'variant':variant,'paper_id':pid,'document_id':did,
                        'source_question_id':target['question_id'],'aligned':alignments[target['question_id']],
                        'branch_ranks':{m:branches[m][did] for m in branches},'fused_ids':[c['chunk_id'] for c in fused],
                        'stage_ids':{'corpus':[c['chunk_id'] for c in members[pid].chunks],
                            'candidates':[c['chunk_id'] for c in fused],'paper_top5':[c['chunk_id'] for c in fused[:5]],
                            'selected_top5':[c['chunk_id'] for c in selected if c['paper_id']==did],
                            'context':[s['chunk_id'] for s in sources if s['paper_id']==did]}}
                    traces.append(trace);probes.append(probe_evidence(members[pid].chunks,trace))
    finally:store.close()
    for name,h in source_hashes.items():
        if sha256((ROOT/name).read_bytes())!=h:raise ValueError('Code changed')
    write_new(run/'chunks.json',saved_chunks);write_new(run/'traces.json',traces)
    report={'scope':'Offline first-loss diagnostic on fixed development gold evidence-unit occurrences, not answer quality or causal proof.',
        'run_id':run.name,'query_run_id':args.query_run.name,'generation_api_calls':0,'question_count':len(cases),
        'source_code_sha256':source_hashes,'comparison_sha256':sha256((args.query_run/'comparison.json').read_bytes()),
        'artifacts_sha256':{n:sha256((run/n).read_bytes()) for n in ['chunks.json','traces.json']},
        'variants':summarize_probes(probes),'records':probes,
        'stage_definitions':{'alignment_or_chunking':'Full corpus does not contain the aligned gold unit.',
            'candidate_retrieval':'Branch candidate union loses full gold coverage.',
            'fusion_ranking':'Per-paper RRF top five loses full coverage.',
            'document_quota':'Multi-document five-anchor allocation loses full coverage.',
            'context_budget':'7400-character rendering loses full coverage.',
            'covered':'Gold unit remains fully present in rendered evidence.'},
        'note':'Stages are nested; counts show the first full-coverage loss. Overlap ranks are not semantic relevance scores. Query correctness requires review. No production change.'}
    write_new(run/'summary.json',report);write_new(args.output,report)
    print({'run':str(run),'variants':report['variants']})


if __name__=='__main__':main()
