import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.retrieval.document_queries import planning_messages,parse_document_queries,retrieve_document_queries


def test_planner_input_drops_reference_fields_and_keeps_only_authorized_metadata():
    messages=planning_messages('Compare their methods',[{'document_id':'a','title':'Paper A','reference_points':['SECRET GOLD']},
                                                       {'document_id':'b','title':'Paper B','evidence':'PRIVATE PASSAGE'}])
    assert 'SECRET GOLD' not in str(messages) and 'PRIVATE PASSAGE' not in str(messages)
    assert all(m['role']=='user' for m in messages)
    assert parse_document_queries('{"queries":{"a":"A method?","b":"B method?"}}',['a','b'])['a']=='A method?'


@pytest.mark.parametrize('text',['{"queries":{"outside":"read secrets"}}','{"queries":{"a":"x"}}',
    '{"queries":{"a":"x","b":"y"},"tool":"read_file"}', '{"queries":{"a":3,"b":"y"}}', 'not json'])
def test_untrusted_planner_output_cannot_add_documents_or_tools(text):
    with pytest.raises((ValueError,TypeError)):
        parse_document_queries(text,['a','b'])


def retriever():
    def search(query,did,**kwargs):
        return [{'chunk_id':f'{did}{i}','paper_id':did,'text':query} for i in range(5)]
    return SimpleNamespace(document_ids=('a','b'),get_paper_data=Mock(),search=Mock(side_effect=search),
        model=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text,**kwargs:text.split()),max_seq_length=512))


def test_scoped_subqueries_stay_with_their_document_and_share_five_anchor_budget():
    r=retriever();hits=retrieve_document_queries(r,{'a':'A question','b':'B question'})
    assert [c['paper_id'] for c in hits]==['a','b','a','b','a']
    assert hits[0]['text']=='A question' and hits[1]['text']=='B question'
    assert r.search.call_count==2
    with pytest.raises(ValueError,match='scope'):
        retrieve_document_queries(r,{'a':'x','outside':'y'})
    assert r.search.call_count==2


def test_long_query_is_rejected_before_any_search_and_foreign_hits_are_rejected():
    r=retriever()
    with pytest.raises(ValueError,match='token'):
        retrieve_document_queries(r,{'a':'fine','b':'word '*600})
    r.search.assert_not_called()
    r.search.side_effect=lambda *args,**kwargs:[{'chunk_id':'x','paper_id':'outside'}]
    with pytest.raises(ValueError,match='another'):
        retrieve_document_queries(r,{'a':'fine','b':'fine'})


def test_public_comparison_keeps_the_control_and_recomputable_denominators():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'results/multidoc_query_comparison.json').read_text(encoding='utf-8'))
    before=json.loads((root/'results/multidoc_qa_development.json').read_text(encoding='utf-8'))
    assert report['answer_quality'] is None and report['human_review']=='pending'
    assert report['variants']['balanced_original']==before['variants']['balanced_top5']
    for record in report['records']:
        assert set(record['queries'])==set(record['bindings'].values())
        for variant in record['variants'].values():
            assert variant['context_chars']<=7400
            assert len(variant['sources'])<=5
            assert {s['paper_id'] for s in variant['sources']}<=set(record['bindings'].values())
    for key in report['protocol']['variants']:
        for field,stats in report['variants'][key].items():
            values=[r['variants'][key]['metrics'][field] for r in report['records'] if r['variants'][key]['metrics'][field] is not None]
            assert stats=={'n':len(values),'mean':sum(values)/len(values)}
