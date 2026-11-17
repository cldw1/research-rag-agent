"""Frozen multi-document answer inputs, source-review material and replay."""

from pathlib import Path
import csv
import json
from collections import Counter

from scripts.check_branch_heads import check_head_run
from src.agent.context_window import ByteTokenEstimator
from src.evaluation.pdf_answers import read_json,collect_results,summarize_answers
from src.evaluation.pdf_benchmark import sha256,verify_sources
from src.evaluation.multidoc_qa import resolve_cases
from src.evaluation.answer_review import summarize_reviews, validate_reviews


def generation_summary(results, manifest):
    """Keep actual usage, without reusing another experiment's price assumptions."""
    for result in results:
        usage = result.get('usage', {})
        if any(type(value) is not int or value < 0 for value in usage.values()):
            raise ValueError('Invalid provider token usage')
        if all(key in usage for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
            if usage['prompt_tokens'] + usage['completion_tokens'] != usage['total_tokens']:
                raise ValueError('Inconsistent provider token accounting')
    summary = summarize_answers(results, manifest)
    summary.pop('estimated_returned_cost_usd_upper_at_published_peak_rates')
    summary['cost_note'] = 'Actual returned tokens only; no price assumption or account bill.'
    summary['cost_usd'] = None
    return summary


def write_human_review_sheet(path, pack, reviews):
    """Export source material for independent review; human columns stay blank."""
    by_id = {row['question_id']: row for row in reviews['records']}
    fields = ['question_id', 'comparison_question_id', 'variant', 'question', 'answerable',
              'references', 'provided_sources', 'gold_evidence', 'answer', 'answer_sha256',
              'ai_correctness_0_2', 'ai_faithfulness_0_2', 'ai_citation_support_0_2',
              'ai_citation_coverage_0_2', 'ai_paper_attribution_0_2', 'ai_reason',
              'human_correctness_0_2', 'human_faithfulness_0_2', 'human_citation_support_0_2',
              'human_citation_coverage_0_2', 'human_paper_attribution_0_2',
              'human_reviewer', 'human_reviewed_at', 'human_notes']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in pack:
            ai = by_id[row['question_id']]
            values = {key: row[key] for key in fields[:10]}
            for key in ('references', 'provided_sources', 'gold_evidence'):
                values[key] = json.dumps(values[key], ensure_ascii=False)
            for key in ('correctness', 'faithfulness', 'citation_support', 'citation_coverage', 'paper_attribution'):
                values['ai_' + key + '_0_2'] = ai[key]
            values['ai_reason'] = ai['reason']
            writer.writerow(values)


def load_requests(root):
    root=Path(root);protocol=read_json(root/'benchmarks/multidoc_qa/answer_protocol.json')
    head=root/'data/pdf_benchmark/head_runs'/protocol['head_run'];report=check_head_run(head)
    prompts={(p['question_id'],p['variant']):p for p in read_json(head/'answer_prompts.json')}
    requests=[]
    for row in report['records']:
        qid=row['question_id']
        for variant in protocol['variants']:
            p=prompts[(qid,variant)];messages=[{'role':'user','content':p['prompt']}]
            requests.append({'question_id':qid+'--'+variant,'comparison_question_id':qid,'variant':variant,
                'paper_id':'selected_documents','messages':messages,'sources':p['sources'],
                'prompt_sha256':sha256(p['prompt'].encode()),'input_estimate':ByteTokenEstimator().count(messages)})
    if len(requests)!=24 or protocol['variants']!=['rrf','bm25_first']:raise ValueError('Unexpected fixed answer population')
    return protocol,requests


def verify_run(run,root):
    run,root=Path(run),Path(root);manifest=read_json(run/'manifest.json')
    for name,h in manifest['inputs_sha256'].items():
        if sha256((root/name).read_bytes())!=h:raise ValueError('Frozen answer inputs changed')
    for name,h in manifest['source_code_sha256'].items():
        p=(run/'source_snapshot'/name).resolve()
        if (run/'source_snapshot').resolve() not in p.parents or sha256(p.read_bytes())!=h:raise ValueError('Answer snapshot changed')
    if sha256((run/'requests.json').read_bytes())!=manifest['requests_file_sha256']:raise ValueError('Answer requests changed')
    _,expected=load_requests(root);requests=read_json(run/'requests.json')
    if requests!=expected:raise ValueError('Answer prompts differ from frozen retrieval')
    return manifest,requests


def make_review_pack(run,root):
    run,root=Path(run),Path(root);manifest,requests=verify_run(run,root)
    retrieval=root/'data/pdf_benchmark/runs/20261115T165227Z-1bd0349b'
    dataset=read_json(retrieval/'dataset.json');native,_=verify_sources(dataset,root/'data/pdf_benchmark/sources')
    cases={c['question_id']:c for c in resolve_cases(read_json(root/'benchmarks/multidoc_qa/questions.json'),dataset)}
    chunks=read_json(retrieval/'chunks.json');lookup={c['chunk_id']:c for rows in chunks.values() for c in rows}
    aliases={c['paper_id']:pid for pid,rows in chunks.items() for c in rows}
    output=[]
    for request,result in zip(requests,collect_results(run,requests),strict=True):
        case=cases[request['comparison_question_id']]
        row={'question_id':request['question_id'],'comparison_question_id':case['question_id'],'variant':request['variant'],
            'question':case['question'],'answerable':any(t['answerable'] for t in case['targets']),
            'references':{t['paper_id']:{'answerable':t['answerable'],'points':t['reference_points']} for t in case['targets']},
            'status':result['status'],'provided_sources':[{'citation':s['citation'],'paper_id':s['paper_id'],
                'paper_alias':aliases[s['paper_id']],'chunk_id':s['chunk_id'],'page_number':s.get('page_number'),
                'text':lookup[s['chunk_id']]['text']} for s in request['sources']],
            'gold_evidence':[]}
        for target in case['targets']:
            for a in target['evidence']:
                row['gold_evidence'].append({'paper_alias':target['paper_id'],'page_number':a['page_number'],
                    'text':native[target['paper_id']][(a['page_number'],a['block_index'])]['text'][a['char_start']:a['char_end']]})
        if result['status'] in ('succeeded','truncated','empty'):
            row.update(answer=result['answer'],answer_sha256=sha256(result['answer'].encode()),
                cited_numbers=result['cited_numbers'],invalid_citations=result['invalid_citations'])
        output.append(row)
    return output


def build_report(run,root,reviews_file=None):
    run,root=Path(run),Path(root);manifest,requests=verify_run(run,root)
    pack=make_review_pack(run,root);results=collect_results(run,requests)
    rubric=read_json(root/'benchmarks/multidoc_qa/answer_rubric.json')
    reviews=read_json(reviews_file) if reviews_file else None
    if reviews:
        if len(reviews['records'])!=24:raise ValueError('Need all 24 source reviews')
        if reviews['reviewer_type'] != 'ai_source_audit' or reviews.get('independent_human_review') is not False:
            raise ValueError('This export requires an explicitly provisional AI audit; human reviews remain separate')
        if not validate_reviews(pack, reviews, rubric)['complete']:
            raise ValueError('Incomplete review')
        for r in reviews['records']:
            if r['paper_attribution'] is not None and (type(r['paper_attribution']) is not int or r['paper_attribution'] not in (0,1,2)):
                raise ValueError('Invalid paper-attribution score')
            if not r.get('paper_attribution_reason','').strip():raise ValueError('Paper-attribution rationale required')
        by_id={r['question_id']:r for r in reviews['records']}
    variants={}
    for variant in ('rrf','bm25_first'):
        chosen=[p for p in pack if p['variant']==variant]
        generated=[r for r,q in zip(results,requests,strict=True) if q['variant']==variant]
        summary=generation_summary(generated,manifest)
        if reviews:
            subset={**reviews,'records':[by_id[p['question_id']] for p in chosen]}
            quality=summarize_reviews(chosen,subset,rubric)
            if not quality['complete']:raise ValueError('Incomplete review')
            values=[r['paper_attribution'] for r in subset['records'] if r['paper_attribution'] is not None]
            quality['paper_attribution']={'applicable':len(values),'scores':{str(k):v for k,v in Counter(values).items()}}
            summary['review_status'] = 'ai_source_audited_human_pending' if reviews['reviewer_type']=='ai_source_audit' else 'human_reviewed'
            summary['ai_source_review']=quality
        variants[variant]=summary
    report={'scope':manifest['scope'],'run_id':run.name,'answer_count':24,'model_config':manifest['model_config'],
        'variants':variants,'total_generation':generation_summary(results,manifest),'independent_human_metrics':None,
        'generation_source_snapshot_verified':True,
        'thinking':manifest['thinking'],
        'observed_response_models':sorted({r.get('response_metadata',{}).get('model') for r in results if r.get('response_metadata',{}).get('model')}),
        'manifest_sha256':sha256((run/'manifest.json').read_bytes()),
        'result_sha256':{r['question_id']:sha256((run/'attempts'/r['question_id']/'result.json').read_bytes()) for r in results if (run/'attempts'/r['question_id']/'result.json').exists()},
        'review_records':reviews['records'] if reviews else None,'review_status':'ai_source_audited_human_pending' if reviews else 'pending'}
    if reviews:
        report['review_provenance']={k:reviews[k] for k in ('rubric_version','reviewer_type','reviewer','reviewed_at','independent_human_review')}
        comparisons=[]
        for qid in dict.fromkeys(p['comparison_question_id'] for p in pack):
            a,b=by_id[qid+'--rrf']['correctness'],by_id[qid+'--bm25_first']['correctness']
            comparisons.append({'question_id':qid,'rrf':a,'bm25_first':b,'difference':b-a})
        report['paired_correctness']={'records':comparisons,'improved':sum(r['difference']>0 for r in comparisons),
            'unchanged':sum(r['difference']==0 for r in comparisons),'regressed':sum(r['difference']<0 for r in comparisons)}
        report.update(reviews_sha256=sha256(Path(reviews_file).read_bytes()),rubric_sha256=sha256((root/'benchmarks/multidoc_qa/answer_rubric.json').read_bytes()))
    return report,pack
