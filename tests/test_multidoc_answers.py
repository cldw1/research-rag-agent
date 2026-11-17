from pathlib import Path
from copy import deepcopy
import csv
import json
import pytest

from src.evaluation import multidoc_answers
from src.evaluation.multidoc_answers import load_requests, generation_summary, write_human_review_sheet

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT/'data/pdf_benchmark/head_runs/20261116T083848Z-5daefc85/summary.json').exists(),reason='Optional local frozen retrieval artifacts')
def test_multidoc_answer_inputs_are_paired_and_exclude_answer_keys():
    protocol,requests=load_requests(ROOT)
    assert len(requests)==24 and len({r['question_id'] for r in requests})==24
    assert protocol['variants']==['rrf','bm25_first']
    for i in range(0,len(requests),2):
        a,b=requests[i:i+2]
        assert a['comparison_question_id']==b['comparison_question_id']
        assert a['variant']=='rrf' and b['variant']=='bm25_first'
        for request in (a,b):
            assert len(request['sources'])<=5
            assert not {'references','reference_points','gold_evidence','answerable'} & set(request)
            assert all(set(m)=={'role','content'} and m['role']=='user' for m in request['messages'])


def public_report():
    return json.loads((ROOT/'results/multidoc_answer_comparison.json').read_text(encoding='utf-8'))


def test_public_answer_comparison_reconciles_both_fixed_populations():
    report=public_report()
    records=report['review_records']
    assert len(records)==24 and len({r['question_id'] for r in records})==24
    assert report['independent_human_metrics'] is None
    assert report['review_provenance']['independent_human_review'] is False
    assert report['review_status']=='ai_source_audited_human_pending'
    assert report['observed_response_models']==[report['model_config']['model']]
    for variant,summary in report['variants'].items():
        selected=[r for r in records if r['question_id'].endswith('--'+variant)]
        assert len(selected)==12 and summary['status_counts']=={'succeeded':12}
        review=summary['ai_source_review']
        assert review['complete'] and review['human_metrics'] is None
        for metric,counts in review['metrics'].items():
            values=[r[metric] for r in selected if r[metric] is not None]
            assert counts['applicable']==len(values)
            for score in range(3):
                assert counts[f'score_{score}']==values.count(score)
        assert sum(review['paper_attribution']['scores'].values())==review['paper_attribution']['applicable']
        assert all(r['paper_attribution_reason'].strip() for r in selected)
    usage=report['total_generation']['known_returned_usage']
    assert usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens']
    for key,total in usage.items():
        assert sum(v['known_returned_usage'][key] for v in report['variants'].values())==total
    paired=report['paired_correctness']
    assert len(paired['records'])==12
    by_id={r['question_id']:r for r in records}
    for r in paired['records']:
        assert r['difference']==by_id[r['question_id']+'--bm25_first']['correctness']-by_id[r['question_id']+'--rrf']['correctness']
    assert paired['improved']==sum(r['difference']>0 for r in paired['records'])
    assert paired['unchanged']==sum(r['difference']==0 for r in paired['records'])
    assert paired['regressed']==sum(r['difference']<0 for r in paired['records'])
    assert set(report['result_sha256'])==set(by_id)
    assert 'estimated_returned_cost_usd_upper_at_published_peak_rates' not in report['total_generation']
    assert not any(key in report for key in ('answer','messages','provided_sources','api_key'))


@pytest.mark.parametrize('usage',[
    {'prompt_tokens':True}, {'completion_tokens':-1},
    {'prompt_tokens':2,'completion_tokens':3,'total_tokens':6},
])
def test_rejects_invalid_provider_usage_before_export(usage):
    with pytest.raises(ValueError,match='token'):
        generation_summary([{'usage':usage}],{})


@pytest.mark.parametrize('field,value,match',[
    ('paper_attribution',True,'paper-attribution'),
    ('paper_attribution',3,'paper-attribution'),
    ('paper_attribution_reason',' ','rationale'),
    ('answer_sha256','changed','changed'),
    ('checked_citations',[],'Citation audit'),
])
def test_answer_export_rejects_changed_or_incomplete_audit(monkeypatch, tmp_path, field, value, match):
    report=public_report()
    reviews={**report['review_provenance'],'records':deepcopy(report['review_records'])}
    pack=[{'question_id':r['question_id'],'answer_sha256':r['answer_sha256'],
           'cited_numbers':r['checked_citations']} for r in report['review_records']]
    reviews['records'][0][field]=value
    path=tmp_path/'reviews.json'
    path.write_text(json.dumps(reviews),encoding='utf-8')
    monkeypatch.setattr(multidoc_answers,'verify_run',lambda *args: ({},[]))
    monkeypatch.setattr(multidoc_answers,'make_review_pack',lambda *args: pack)
    monkeypatch.setattr(multidoc_answers,'collect_results',lambda *args: [])
    with pytest.raises(ValueError,match=match):
        multidoc_answers.build_report(tmp_path,ROOT,path)


def test_human_review_sheet_keeps_scores_blank_and_preserves_sources(tmp_path):
    ai=public_report()['review_records'][0]
    row={'question_id':ai['question_id'],'comparison_question_id':'md-01','variant':'rrf',
         'question':'Compare the papers.','answerable':True,'references':{'a':{'points':['one']}},
         'provided_sources':[{'citation':1,'paper_alias':'a','page_number':2,'text':'source text'}],
         'gold_evidence':[{'paper_alias':'a','text':'gold text'}],
         'answer':'Example answer','answer_sha256':ai['answer_sha256']}
    path=tmp_path/'human.csv'
    write_human_review_sheet(path,[row],{'records':[ai]})
    with path.open(encoding='utf-8-sig',newline='') as stream:
        result=list(csv.DictReader(stream))[0]
    assert all(not value for key,value in result.items() if key.startswith('human_'))
    assert json.loads(result['provided_sources'])==row['provided_sources']
    assert result['ai_correctness_0_2']==str(ai['correctness'])
    with pytest.raises(FileExistsError):
        write_human_review_sheet(path,[row],{'records':[ai]})


@pytest.mark.skipif(not (ROOT/'data/pdf_benchmark/multidoc_answer_runs/20261116T124957Z-914040f8/ai_reviews.json').exists(),reason='Optional local answer artifacts')
def test_actual_answer_comparison_replays_without_model_calls():
    from scripts.check_multidoc_answers import check_answers
    report=public_report()
    run=ROOT/'data/pdf_benchmark/multidoc_answer_runs'/report['run_id']
    result=check_answers(run,ROOT/'results/multidoc_answer_comparison.json')
    assert result['report_verified'] and result['answer_count']==24
    assert result['model_calls']==0
