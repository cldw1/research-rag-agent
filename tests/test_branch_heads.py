from copy import deepcopy

import pytest

from src.retrieval.branch_heads import select_branch_heads


def fixture(ids=('a','b')):
    fused={did:[{'chunk_id':did+str(i),'paper_id':did} for i in (2,0,1)] for did in ids}
    heads={did:{'bm25':did+'0','dense':did+'1'} for did in ids}
    return fused,heads


@pytest.mark.parametrize('first,expected',[('bm25',['a0','b0','a1','b1','a2']),('dense',['a1','b1','a0','b0','a2'])])
def test_first_branch_order_is_explicit_with_fixed_five_slots(first,expected):
    fused,heads=fixture();before=deepcopy((fused,heads))
    result=select_branch_heads(('a','b'),fused,heads,first_branch=first)
    assert [c['chunk_id'] for c in result]==expected
    assert (fused,heads)==before


def test_repeated_or_empty_branch_heads_do_not_duplicate_sources():
    fused,heads=fixture();heads['a']['dense']='a0';heads['b']={'bm25':None,'dense':None}
    result=select_branch_heads(('a','b'),fused,heads)
    assert len({c['chunk_id'] for c in result})==len(result)==5
    assert result[1]['chunk_id']=='b2'


def test_three_papers_cannot_force_six_heads_into_five_slots():
    ids=('a','b','c');fused,heads=fixture(ids)
    result=select_branch_heads(ids,fused,heads)
    assert [c['chunk_id'] for c in result]==['a0','b0','c0','a1','b1']


def test_foreign_or_unregistered_heads_and_duplicated_chunk_ids_are_rejected():
    fused,heads=fixture();heads['a']['bm25']='private-other'
    with pytest.raises(ValueError,match='outside'):select_branch_heads(('a','b'),fused,heads)
    fused,heads=fixture();fused['a'][0]['paper_id']='b'
    with pytest.raises(ValueError,match='identity'):select_branch_heads(('a','b'),fused,heads)


@pytest.mark.parametrize('count',[0,-1,True])
def test_invalid_anchor_limits_are_not_silently_accepted(count):
    fused,heads=fixture()
    with pytest.raises(ValueError,match='budget'):select_branch_heads(('a','b'),fused,heads,top_k=count)


def test_public_head_comparison_reconciles_control_changes_and_limits():
    from pathlib import Path
    from src.evaluation.pdf_answers import read_json
    root=Path(__file__).resolve().parents[1]
    report=read_json(root/'results/multidoc_head_comparison.json')
    old=read_json(root/'results/multidoc_query_comparison.json')
    assert report['variants']['rrf']==old['variants']['balanced_original']
    assert report['generation_api_calls']==0 and report['answer_quality'] is None
    for row in report['records']:
        for v in row['variants'].values():
            assert len(v['selected_chunk_ids'])<=5 and len(set(v['selected_chunk_ids']))==len(v['selected_chunk_ids'])
            assert v['context_chars']<=7400
            assert v['heads_in_context']<=v['heads_selected']<=v['unique_branch_heads']
    for key,counts in report['changes'].items():
        pairs=[(r['variants']['rrf']['metrics']['mean_full_evidence_recall'],r['variants'][key]['metrics']['mean_full_evidence_recall']) for r in report['records'] if r['variants']['rrf']['metrics']['mean_full_evidence_recall'] is not None]
        assert counts=={'improved':sum(b>a for a,b in pairs),'unchanged':sum(b==a for a,b in pairs),'worsened':sum(b<a for a,b in pairs)}
