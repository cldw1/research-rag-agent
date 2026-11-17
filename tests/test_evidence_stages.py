from copy import deepcopy

import pytest

from src.evaluation.evidence_stages import probe_evidence,summarize_probes


def fixture():
    chunks=[{'chunk_id':'good','paper_id':'A','source_metadata':{'spans':[{'block_id':'b','char_start':0,'char_end':10}]}},
            {'chunk_id':'bad','paper_id':'A','source_metadata':{'spans':[{'block_id':'other','char_start':0,'char_end':10}]}}]
    trace={'question_id':'q','variant':'v','paper_id':'a','document_id':'A','source_question_id':'a1',
        'aligned':[{'evidence_id':'e1','status':'mapped','span':['b',0,10]}],
        'branch_ranks':{'bm25':[{'chunk_id':'good','rank':2},{'chunk_id':'bad','rank':1}]},
        'fused_ids':['bad','good'],
        'stage_ids':{name:['bad','good'] for name in ['corpus','candidates','paper_top5','selected_top5','context']}}
    return chunks,trace


@pytest.mark.parametrize('first,label',[(None,'covered'),('corpus','alignment_or_chunking'),
    ('candidates','candidate_retrieval'),('paper_top5','fusion_ranking'),
    ('selected_top5','document_quota'),('context','context_budget')])
def test_first_loss_is_exclusive_and_context_loss_is_not_called_candidate_failure(first,label):
    chunks,trace=fixture()
    if first=='corpus':trace['aligned'][0]['span']=['not-present',0,10]
    elif first:
        after=False
        for name in trace['stage_ids']:
            after=after or name==first
            if after:trace['stage_ids'][name]=['bad']
        if first=='candidates':trace['fused_ids']=['bad']
    row=probe_evidence(chunks,trace)
    assert row['units'][0]['first_loss']==label
    assert row['units'][0]['first_overlap_branch_rank']['bm25']==(None if first=='corpus' else 2)
    counts=summarize_probes([row])['v']['first_loss_counts']
    assert counts=={label:1}


def test_partial_overlap_does_not_count_as_complete_evidence():
    chunks,trace=fixture()
    chunks[0]['source_metadata']['spans'][0]['char_end']=5
    row=probe_evidence(chunks,trace)['units'][0]
    assert row['first_loss']=='alignment_or_chunking'
    assert row['coverage']['corpus']['evidence_recall']==1
    assert row['full_coverage_within_paper_rank'] is None


def test_cross_paper_collisions_and_non_nested_stage_sets_are_rejected():
    chunks,trace=fixture();foreign=deepcopy(chunks)
    foreign[0]['paper_id']='B'
    with pytest.raises(ValueError,match='namespace'):probe_evidence(foreign,trace)
    trace['stage_ids']['candidates']=['bad'];trace['fused_ids']=['bad']
    with pytest.raises(ValueError,match='nested'):probe_evidence(chunks,trace)


def test_public_stage_counts_reconcile_with_every_evidence_unit_occurrence():
    from pathlib import Path
    from src.evaluation.pdf_answers import read_json
    root=Path(__file__).resolve().parents[1]
    report=read_json(root/'results/multidoc_stage_audit.json')
    assert report['generation_api_calls']==0
    assert summarize_probes(report['records'])==report['variants']
    for stats in report['variants'].values():
        assert sum(stats['first_loss_counts'].values())==stats['evidence_unit_occurrences']
    for record in report['records']:
        for unit in record['units']:
            if unit['first_loss']=='covered':assert unit['coverage']['context']['full_evidence_recall']==1
