"""Compare unified, per-paper original and per-paper planned retrieval offline."""

import argparse
from pathlib import Path
import os
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.check_multidoc_results import check_run
from src.evaluation.pdf_answers import read_json,write_new,collect_results,summarize_answers
from src.evaluation.pdf_benchmark import sha256
from src.evaluation.multidoc_qa import resolve_cases,score_context
from src.retrieval.document_queries import parse_document_queries,retrieve_document_queries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    if args.output.exists(): raise FileExistsError('Result already exists')
    manifest=read_json(run/'manifest.json')
    for name,h in manifest['inputs_sha256'].items():
        if sha256((ROOT/name).read_bytes())!=h: raise ValueError('Frozen experiment input changed')
    for name,h in manifest['source_code_sha256'].items():
        if sha256((run/'source_snapshot'/name).read_bytes())!=h: raise ValueError('Source snapshot changed')
    if sha256((run/'requests.json').read_bytes())!=manifest['requests_file_sha256']: raise ValueError('Planner inputs changed')
    protocol=read_json(ROOT/'benchmarks/multidoc_qa/query_protocol.json')
    retrieval=ROOT/'data/pdf_benchmark/runs'/protocol['retrieval_run']
    baseline=check_run(ROOT/'data/pdf_benchmark/multidoc_runs'/protocol['baseline_run'],retrieval)
    dataset=read_json(retrieval/'dataset.json')
    cases=resolve_cases(read_json(ROOT/'benchmarks/multidoc_qa/questions.json'),dataset)
    requests=read_json(run/'requests.json');results=collect_results(run,requests)
    if any(r['status']=='pending' for r in results): raise ValueError('Planner population not completed')
    result_by_id={r['question_id']:r for r in results}
    alignments={}
    for a in read_json(retrieval/'alignments.json'): alignments.setdefault(a['question_id'],[]).append(a)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from src.embedding import load_embedding_resources
    from src.storage.document_store import LocalDocumentStore
    from src.retrieval.multi_document_retriever import MultiDocumentRetriever
    from src.generation.strict_context_builder import build_strict_context
    torch.set_num_threads(8)
    model,_,profile=load_embedding_resources(local_files_only=True)
    if profile.to_dict()!=baseline['embedding_profile']: raise ValueError('Embedding changed')
    if not (run/'catalog').exists(): shutil.copytree(retrieval/'catalog',run/'catalog')
    store=LocalDocumentStore(run/'catalog');members={};records=[];prompts=[]
    try:
        for b in read_json(retrieval/'summary.json')['builds']:
            members[b['paper_id']]=store.load_retriever(b['document_id'],model,profile)
        for case in cases:
            pids=[t['paper_id'] for t in case['targets']]
            retriever=MultiDocumentRetriever([members[p] for p in pids]);bindings={p:members[p].paper_id for p in pids}
            original={did:case['question'] for did in retriever.document_ids}
            status='valid';result=result_by_id[case['question_id']]
            try:
                if result['status']!='succeeded': raise ValueError('Planner not completed')
                queries=parse_document_queries(result['answer'],retriever.document_ids)
                from src.retrieval.uploaded_pdf_retriever import QUERY_PREFIX
                if any(len(model.tokenizer.encode(QUERY_PREFIX+q,add_special_tokens=True,truncation=False))>model.max_seq_length for q in queries.values()):
                    raise ValueError('Long query')
            except (ValueError,TypeError,KeyError): queries=original;status='fallback_original'
            variants={}
            for key in protocol['variants']:
                start=time.perf_counter()
                hits=retriever.search(case['question'],retriever.paper_id,top_k=5,candidate_k=50) if key=='balanced_original' else retrieve_document_queries(retriever,original if key=='scoped_original' else queries)
                seconds=time.perf_counter()-start
                hits=[{**c,'persistent_index_id':retriever.document_bindings[c['paper_id']]} for c in hits]
                context,sources=build_strict_context(hits,max_chars=7400)
                if key=='balanced_original':
                    old=next(r for r in baseline['records'] if r['question_id']==case['question_id'])['variants']['balanced_top5']
                    if sources!=old['sources']: raise ValueError('Baseline did not reproduce')
                variants[key]={'sources':sources,'context_chars':len(context),'seconds':seconds,
                    'retrieval_calls':1 if key=='balanced_original' else len(pids),
                    'metrics':score_context(sources,retriever.chunks,case['targets'],bindings,alignments)}
                from src.generation.context_builder import build_qa_prompt
                prompts.append({'question_id':case['question_id'],'variant':key,'prompt':build_qa_prompt(case['question'],context),'sources':sources})
            records.append({'question_id':case['question_id'],'bindings':bindings,'queries':queries,'planner_status':status,'variants':variants})
    finally: store.close()
    aggregate={}
    for key in protocol['variants']:
        aggregate[key]={}
        for field in ['document_presence','mean_full_evidence_recall','all_required_papers_hit','all_gold_complete']:
            values=[r['variants'][key]['metrics'][field] for r in records if r['variants'][key]['metrics'][field] is not None]
            aggregate[key][field]={'n':len(values),'mean':sum(values)/len(values) if values else None}
    write_new(run/'retrieval_records.json',records);write_new(run/'answer_prompts.json',prompts)
    report={'scope':protocol['scope'],'run_id':run.name,'question_count':len(records),'variants':aggregate,
        'protocol':protocol,'planner_generation':summarize_answers(results,manifest),'answer_quality':None,'human_review':'pending',
        'planner_fallbacks':sum(r['planner_status']!='valid' for r in records),'records':records,
        'manifest_sha256':sha256((run/'manifest.json').read_bytes()),
        'artifacts_sha256':{n:sha256((run/n).read_bytes()) for n in ['requests.json','retrieval_records.json','answer_prompts.json']},
        'planner_result_sha256':{r['question_id']:sha256((run/'attempts'/r['question_id']/'result.json').read_bytes()) for r in results if (run/'attempts'/r['question_id']/'result.json').exists()}}
    write_new(run/'comparison.json',report);write_new(args.output,report)
    print({'run':str(run),'variants':aggregate,'fallbacks':report['planner_fallbacks']})


if __name__=='__main__': main()
