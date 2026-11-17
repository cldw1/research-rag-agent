"""Prepare, execute, or replay a bounded paired answer-policy regression check."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.context_window import ByteTokenEstimator
from src.evaluation.answer_review import summarize_reviews, validate_reviews
from src.evaluation.multidoc_answers import verify_run, make_review_pack, generation_summary, write_human_review_sheet
from src.evaluation.pdf_answers import read_json, write_new, generate_one, collect_results, attempt_paths
from src.evaluation.pdf_benchmark import sha256
from src.generation.answer_policy import ANSWER_POLICY_VERSION, ANSWER_GROUNDING_RULES, build_grounded_qa_prompt
from src.generation.context_builder import build_qa_prompt
from src.llm.provider import ModelConfig, OpenAICompatibleProvider

PROTOCOL = 'benchmarks/multidoc_qa/answer_policy_protocol.json'
SOURCES = ['scripts/run_answer_policy_check.py', 'src/generation/answer_policy.py',
           'src/generation/context_builder.py', 'src/evaluation/pdf_answers.py',
           'src/evaluation/multidoc_answers.py', 'src/llm/provider.py']


def prepare_requests(root=ROOT, policy=None):
    root = Path(root)
    protocol = read_json(root / PROTOCOL)
    source_run = root / 'data/pdf_benchmark/multidoc_answer_runs' / protocol['source_run']
    _, original = verify_run(source_run, root)
    originals = {r['comparison_question_id']: r for r in original if r['variant'] == 'rrf'}
    questions = {r['question_id']: r['question'] for r in read_json(root / 'benchmarks/multidoc_qa/questions.json')['questions']}
    policy = policy or {'version': ANSWER_POLICY_VERSION, 'rules': ANSWER_GROUNDING_RULES}
    requests = []
    for qid in protocol['question_ids']:
        old = originals[qid]
        prompt = old['messages'][0]['content']
        context = prompt.split('RETRIEVED EVIDENCE:\n', 1)[1].rsplit('\n\nANSWER:', 1)[0]
        if prompt != build_qa_prompt(questions[qid], context):
            raise ValueError('Frozen question or evidence differs from the legacy template')
        guarded = f"Answer policy: {policy['version']}\n{policy['rules']}\n\n{prompt}"
        if policy['version'] == ANSWER_POLICY_VERSION and policy['rules'] == ANSWER_GROUNDING_RULES:
            if guarded != build_grounded_qa_prompt(questions[qid], context):
                raise ValueError('Live and regression answer rules differ')
        for variant, text in (('legacy', prompt), ('grounded', guarded)):
            messages = [{'role': 'user', 'content': text}]
            requests.append({'question_id': qid+'--'+variant, 'comparison_question_id': qid,
                'variant': variant, 'paper_id': old['paper_id'], 'sources': old['sources'],
                'messages': messages, 'prompt_sha256': sha256(text.encode()),
                'source_prompt_sha256': old['prompt_sha256'],
                'input_estimate': ByteTokenEstimator().count(messages)})
    if len(requests) != 12 or len({r['question_id'] for r in requests}) != 12 or protocol['max_api_calls'] != 12:
        raise ValueError('Expected six fixed pairs and at most twelve model calls')
    if protocol['variants'] != ['legacy', 'grounded'] or protocol['source_variant'] != 'rrf':
        raise ValueError('Comparison variants changed')
    return protocol, requests, policy


def verify_check(run, root=ROOT):
    run, root = Path(run), Path(root)
    manifest = read_json(run / 'manifest.json')
    for name, expected in manifest['input_sha256'].items():
        if sha256((root / name).read_bytes()) != expected:
            raise ValueError('Comparison input changed')
    for name, expected in manifest['source_sha256'].items():
        path = (run / 'source_snapshot' / name).resolve()
        if (run / 'source_snapshot').resolve() not in path.parents or sha256(path.read_bytes()) != expected:
            raise ValueError('Comparison source snapshot changed')
    for name, expected in manifest['artifact_sha256'].items():
        if sha256((run / name).read_bytes()) != expected:
            raise ValueError('Frozen comparison artifact changed')
    protocol, expected, _ = prepare_requests(root, read_json(run / 'policy.json'))
    requests = read_json(run / 'requests.json')
    if requests != expected:
        raise ValueError('Comparison requests differ from frozen evidence and policy')
    return manifest, protocol, requests


def build_report(run, reviews_file=None, root=ROOT):
    run, root = Path(run), Path(root)
    manifest, protocol, requests = verify_check(run, root)
    source = root / 'data/pdf_benchmark/multidoc_answer_runs' / protocol['source_run']
    originals = {r['comparison_question_id']: r for r in make_review_pack(source, root) if r['variant'] == 'rrf'}
    results = collect_results(run, requests)
    pack = []
    for request, result in zip(requests, results, strict=True):
        if result['question_id'] != request['question_id']:
            raise ValueError('Returned answer identity changed')
        old = originals[request['comparison_question_id']]
        row = {k: v for k, v in old.items() if k not in ('answer', 'answer_sha256', 'cited_numbers', 'invalid_citations')}
        row.update(question_id=request['question_id'], variant=request['variant'], status=result['status'])
        if 'answer' in result:
            row.update(answer=result['answer'], answer_sha256=sha256(result['answer'].encode()),
                       cited_numbers=result['cited_numbers'], invalid_citations=result['invalid_citations'])
        pack.append(row)
    rubric = read_json(run / 'rubric.json')
    reviews = read_json(reviews_file) if reviews_file else None
    if reviews:
        if reviews['reviewer_type'] != 'ai_source_audit' or reviews.get('independent_human_review') is not False:
            raise ValueError('This report requires an explicitly provisional AI source audit')
        if len(reviews['records']) != 12 or not validate_reviews(pack, reviews, rubric)['complete']:
            raise ValueError('Complete twelve-answer review required')
        for row in reviews['records']:
            if type(row['paper_attribution']) is not int or row['paper_attribution'] not in (0, 1, 2):
                raise ValueError('Invalid paper-attribution score')
            if not row.get('paper_attribution_reason', '').strip():
                raise ValueError('Paper-attribution rationale required')
    variants = {}
    for variant in protocol['variants']:
        selected = [p for p in pack if p['variant'] == variant]
        generated = [r for r, q in zip(results, requests, strict=True) if q['variant'] == variant]
        summary = generation_summary(generated, manifest)
        if reviews:
            subset = {**reviews, 'records': [r for r in reviews['records'] if r['question_id'].endswith('--'+variant)]}
            quality = summarize_reviews(selected, subset, rubric)
            values = [r['paper_attribution'] for r in subset['records']]
            quality['paper_attribution'] = {'applicable': len(values), 'scores': {str(k): v for k, v in Counter(values).items()}}
            summary['ai_source_review'] = quality
        variants[variant] = summary
    report = {'schema_version': 'answer-policy-comparison-v1', 'scope': protocol['scope'],
        'run_id': run.name, 'question_ids': protocol['question_ids'], 'answer_count': 12,
        'model_config': manifest['model_config'], 'thinking': protocol['thinking'],
        'policy_version': read_json(run/'policy.json')['version'], 'source_run': protocol['source_run'],
        'answer_policy': read_json(run/'policy.json'),
        'source_variant': 'rrf', 'total_generation': generation_summary(results, manifest), 'variants': variants,
        'independent_human_metrics': None, 'review_status': 'ai_source_audited_human_pending' if reviews else 'pending',
        'review_records': reviews['records'] if reviews else None,
        'observed_response_models': sorted({r.get('response_metadata', {}).get('model') for r in results if r.get('response_metadata', {}).get('model')}),
        'manifest_sha256': sha256((run/'manifest.json').read_bytes()),
        'result_sha256': {r['question_id']: sha256((run/'attempts'/r['question_id']/'result.json').read_bytes())
                          for r in results if (run/'attempts'/r['question_id']/'result.json').exists()}}
    if reviews:
        by_id = {r['question_id']: r for r in reviews['records']}
        paired = {}
        for metric in ('correctness', 'faithfulness', 'citation_support', 'paper_attribution'):
            pairs = [(by_id[q+'--legacy'][metric], by_id[q+'--grounded'][metric]) for q in protocol['question_ids']]
            paired[metric] = {'improved': sum(b>a for a,b in pairs), 'unchanged': sum(b==a for a,b in pairs),
                              'regressed': sum(b<a for a,b in pairs)}
        report.update(paired_changes=paired, reviews_sha256=sha256(Path(reviews_file).read_bytes()),
                      review_provenance={k:v for k,v in reviews.items() if k!='records'})
    return report, pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--use-environment-proxy', action='store_true')
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args()
    if args.check_report:
        if not args.resume or not args.reviews or args.execute or args.output:
            raise ValueError('Offline check requires --resume and --reviews, without --execute or --output')
        report, _ = build_report(args.resume, args.reviews)
        if report != read_json(args.check_report):
            raise ValueError('Saved report differs from verified answers and reviews')
        print({'report_verified': True, 'answer_count': report['answer_count'], 'model_calls': 0})
        return
    if args.output and args.output.exists():
        raise FileExistsError('Output exists; export to a new path')
    if args.resume:
        run = args.resume.resolve()
        manifest, protocol, requests = verify_check(run)
    else:
        protocol, requests, policy = prepare_requests()
        run = ROOT/'data/pdf_benchmark/answer_policy_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        write_new(run/'requests.json', requests)
        write_new(run/'policy.json', policy)
        rubric = read_json(ROOT/'benchmarks/multidoc_qa/answer_rubric.json')
        rubric.update(rubric_version='answer-policy-source-review-v1', scope=protocol['scope'])
        write_new(run/'rubric.json', rubric)
        config = ModelConfig(model=protocol['model'], temperature=protocol['temperature'],
                             max_output_tokens=protocol['max_output_tokens'], supports_tools=False)
        names = [PROTOCOL, 'benchmarks/multidoc_qa/answer_rubric.json', 'benchmarks/multidoc_qa/questions.json',
                 'results/multidoc_answer_comparison.json']
        manifest = {'scope': protocol['scope'], 'model_config': config.to_dict(), 'environment_proxy': args.use_environment_proxy,
                    'input_sha256': {n:sha256((ROOT/n).read_bytes()) for n in names},
                    'source_sha256': {n:sha256((ROOT/n).read_bytes()) for n in SOURCES},
                    'artifact_sha256': {n:sha256((run/n).read_bytes()) for n in ('requests.json','policy.json','rubric.json')}}
        for name in SOURCES:
            path = run/'source_snapshot'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT/name).read_bytes())
        write_new(run/'manifest.json', manifest)
    if args.execute:
        if args.use_environment_proxy != manifest['environment_proxy']:
            raise ValueError('Transport configuration changed')
        for name, expected in manifest['source_sha256'].items():
            if sha256((ROOT/name).read_bytes()) != expected:
                raise ValueError('Execution code changed')
        pending = [r for r in requests if not attempt_paths(run, r['question_id'])[0].exists()]
        if pending:
            from dotenv import load_dotenv
            from openai import OpenAI, DefaultHttpxClient
            load_dotenv(ROOT/'.env', override=False)
            key = os.getenv('DEEPSEEK_API_KEY', '').strip()
            if not key:
                raise ValueError('Configure the existing DEEPSEEK_API_KEY locally')
            config = ModelConfig(**manifest['model_config'])
            client = OpenAI(api_key=key, base_url=config.base_url, max_retries=0, timeout=60.,
                http_client=DefaultHttpxClient(trust_env=args.use_environment_proxy, follow_redirects=False))
            try:
                provider = OpenAICompatibleProvider(config, key, client=client)
                first = generate_one(run, pending.pop(0), provider)
                print({'question_id':first['question_id'], 'status':first['status']}, flush=True)
                if first['status'] == 'succeeded':
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        for future in as_completed([pool.submit(generate_one, run, r, provider) for r in pending]):
                            result = future.result()
                            print({'question_id':result['question_id'], 'status':result['status']}, flush=True)
            finally:
                client.close()
    report, pack = build_report(run, args.reviews)
    write_new(run/('summary-'+uuid.uuid4().hex[:8]+'.json'), report)
    write_new(run/('review_pack-'+uuid.uuid4().hex[:8]+'.json'), pack)
    if args.output:
        if not args.reviews or any(p['status']!='succeeded' for p in pack):
            raise ValueError('Public export requires all twelve successful answers and complete source review')
        write_new(args.output, report)
        sheet = run/'human_answer_policy_review.csv'
        if not sheet.exists():
            write_human_review_sheet(sheet, pack, read_json(args.reviews))
    print({'run': str(run), 'status_counts':report['total_generation']['status_counts']}, flush=True)


if __name__ == '__main__':
    main()
