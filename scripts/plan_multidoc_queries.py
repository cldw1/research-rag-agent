"""Prepare or execute at most twelve question-only DeepSeek planning requests."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.agent.context_window import ByteTokenEstimator
from src.evaluation.pdf_answers import read_json,write_new,generate_one,attempt_paths,collect_results,summarize_answers
from src.evaluation.pdf_benchmark import sha256,verify_candidate_release
from src.llm.provider import ModelConfig,OpenAICompatibleProvider
from src.retrieval.document_queries import planning_messages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--use-environment-proxy',action='store_true')
    args=parser.parse_args()
    verify_candidate_release(ROOT/'benchmarks/pdf_qa')
    config=ModelConfig(model='deepseek-flash',temperature=0.,max_output_tokens=800,supports_tools=False)
    if args.resume:
        run=args.resume.resolve(); manifest=read_json(run/'manifest.json')
        for name,h in manifest['inputs_sha256'].items():
            if sha256((ROOT/name).read_bytes())!=h: raise ValueError('Frozen planner input changed')
        for name,h in manifest['source_code_sha256'].items():
            if sha256((ROOT/name).read_bytes())!=h: raise ValueError('Planner execution code changed')
        requests=read_json(run/'requests.json')
        if sha256((run/'requests.json').read_bytes())!=manifest['requests_file_sha256']: raise ValueError('Requests changed')
        if config.to_dict()!=manifest['model_config']: raise ValueError('Planner configuration changed')
    else:
        dataset=read_json(ROOT/'benchmarks/pdf_qa/development.json')
        if any(p['split']!='dev' for p in dataset['papers']): raise ValueError('Development only')
        papers={p['paper_id']:p for p in dataset['papers']}
        # Only document membership is derived from the evaluation linkage. No
        # original single-paper question, reference or evidence enters planning.
        membership={q['question_id']:q['paper_id'] for q in dataset['questions']}
        spec=read_json(ROOT/'benchmarks/multidoc_qa/questions.json')
        builds=read_json(ROOT/'data/pdf_benchmark/runs/20261115T165227Z-1bd0349b/summary.json')['builds']
        ids={b['paper_id']:b['document_id'] for b in builds}
        run=ROOT/'data/pdf_benchmark/query_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        requests=[]
        for case in spec['questions']:
            pids=[membership[q] for q in case['source_question_ids']]
            documents=[{'document_id':ids[p],'title':papers[p].get('title',p)} for p in pids]
            messages=planning_messages(case['question'],documents)
            requests.append({'question_id':case['question_id'],'paper_id':'selected_documents','messages':messages,
                'document_ids':[d['document_id'] for d in documents], 'input_estimate':ByteTokenEstimator().count(messages),
                'prompt_sha256':sha256(messages[0]['content'].encode()),'sources':[]})
        if len(requests)!=12: raise ValueError('Expected fixed twelve-question population')
        write_new(run/'requests.json',requests)
        files=['benchmarks/multidoc_qa/questions.json','benchmarks/multidoc_qa/query_protocol.json',
            'benchmarks/pdf_qa/development.json','results/multidoc_qa_development.json',
            'data/pdf_benchmark/runs/20261115T165227Z-1bd0349b/summary.json']
        source_files=['scripts/plan_multidoc_queries.py','scripts/compare_multidoc_queries.py',
            'src/retrieval/document_queries.py','src/evaluation/pdf_answers.py','src/llm/provider.py']
        manifest={'model_config':config.to_dict(),'environment_proxy':args.use_environment_proxy,
            'inputs_sha256':{n:sha256((ROOT/n).read_bytes()) for n in files},
            'source_code_sha256':{n:sha256((ROOT/n).read_bytes()) for n in source_files},
            'requests_file_sha256':sha256((run/'requests.json').read_bytes())}
        for name in source_files:
            p=run/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
        write_new(run/'manifest.json',manifest)
    if args.execute:
        if manifest['environment_proxy']!=args.use_environment_proxy: raise ValueError('Proxy setting changed')
        pending=[r for r in requests if not attempt_paths(run,r['question_id'])[0].exists()]
        if pending:
            from dotenv import load_dotenv
            from openai import OpenAI,DefaultHttpxClient
            load_dotenv(ROOT/'.env',override=False)
            key=os.getenv('DEEPSEEK_API_KEY','').strip()
            if not key or any(s in key.lower() for s in ['your_','dummy','placeholder']): raise ValueError('Configure DEEPSEEK_API_KEY locally')
            client=OpenAI(api_key=key,base_url=config.base_url,max_retries=0,timeout=60.,
                http_client=DefaultHttpxClient(trust_env=args.use_environment_proxy,follow_redirects=False))
            try:
                provider=OpenAICompatibleProvider(config,key,client=client)
                first=generate_one(run,pending.pop(0),provider)
                print({'question_id':first['question_id'],'status':first['status']},flush=True)
                if first['status']=='succeeded':
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        for future in as_completed([pool.submit(generate_one,run,r,provider) for r in pending]):
                            result=future.result();print({'question_id':result['question_id'],'status':result['status']},flush=True)
            finally: client.close()
    summary=summarize_answers(collect_results(run,requests),manifest)
    write_new(run/('planner_summary-'+uuid.uuid4().hex[:8]+'.json'),summary)
    print({'run':str(run),'statuses':summary['status_counts']},flush=True)


if __name__=='__main__': main()
