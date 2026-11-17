"""Prepare or execute 24 frozen public-development RAG responses, with durable intents."""

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import os
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.pdf_answers import read_json,write_new,generate_one,attempt_paths
from src.evaluation.pdf_benchmark import sha256
from src.evaluation.multidoc_answers import load_requests,verify_run,build_report
from src.llm.provider import ModelConfig,OpenAICompatibleProvider


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--use-environment-proxy',action='store_true');parser.add_argument('--reviews',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config=ModelConfig(model='deepseek-flash',temperature=0.,max_output_tokens=1200,supports_tools=False)
    if args.resume:
        run=args.resume.resolve();manifest,requests=verify_run(run,ROOT)
    else:
        protocol,requests=load_requests(ROOT)
        run=ROOT/'data/pdf_benchmark/multidoc_answer_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        write_new(run/'requests.json',requests)
        files=['benchmarks/multidoc_qa/answer_protocol.json','benchmarks/multidoc_qa/answer_rubric.json',
            'benchmarks/multidoc_qa/questions.json','benchmarks/pdf_qa/development.json','results/multidoc_head_comparison.json']
        names=['scripts/run_multidoc_answers.py','src/evaluation/multidoc_answers.py',
            'src/evaluation/pdf_answers.py','src/llm/provider.py','src/generation/context_builder.py']
        manifest={'scope':protocol['scope'],'model_config':config.to_dict(),'thinking':'disabled',
            'environment_proxy':args.use_environment_proxy,'requests_file_sha256':sha256((run/'requests.json').read_bytes()),
            'inputs_sha256':{n:sha256((ROOT/n).read_bytes()) for n in files},
            'source_code_sha256':{n:sha256((ROOT/n).read_bytes()) for n in names}}
        for name in names:
            p=run/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
        write_new(run/'manifest.json',manifest)
    if args.execute:
        if config.to_dict()!=manifest['model_config'] or args.use_environment_proxy!=manifest['environment_proxy']:raise ValueError('Generation configuration changed')
        for name,h in manifest['source_code_sha256'].items():
            if sha256((ROOT/name).read_bytes())!=h:raise ValueError('Execution code changed')
        pending=[r for r in requests if not attempt_paths(run,r['question_id'])[0].exists()]
        if pending:
            from dotenv import load_dotenv
            from openai import OpenAI,DefaultHttpxClient
            load_dotenv(ROOT/'.env',override=False);key=os.getenv('DEEPSEEK_API_KEY','').strip()
            if not key or any(s in key.lower() for s in ['your_','dummy','placeholder']):raise ValueError('Configure the existing DEEPSEEK_API_KEY locally')
            client=OpenAI(api_key=key,base_url=config.base_url,max_retries=0,timeout=60.,
                http_client=DefaultHttpxClient(trust_env=args.use_environment_proxy,follow_redirects=False))
            try:
                provider=OpenAICompatibleProvider(config,key,client=client)
                first=generate_one(run,pending.pop(0),provider)
                print({'question_id':first['question_id'],'status':first['status']},flush=True)
                if first['status']=='succeeded':
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        for future in as_completed([pool.submit(generate_one,run,r,provider) for r in pending]):
                            r=future.result();print({'question_id':r['question_id'],'status':r['status']},flush=True)
            finally:client.close()
    report,pack=build_report(run,ROOT,args.reviews)
    write_new(run/('summary-'+uuid.uuid4().hex[:8]+'.json'),report)
    write_new(run/('review_pack-'+uuid.uuid4().hex[:8]+'.json'),pack)
    if args.output:
        if not args.reviews or any(p['status']!='succeeded' for p in pack):raise ValueError('Public export requires full generation and source review')
        write_new(args.output,report)
    print({'run':str(run),'status_counts':report['total_generation']['status_counts']},flush=True)


if __name__=='__main__':main()
