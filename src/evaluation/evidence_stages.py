"""Diagnose the first loss of a complete gold unit along nested retrieval sets."""

from collections import Counter
from src.evaluation.pdf_benchmark import evidence_coverage


STAGES = ('corpus','candidates','paper_top5','selected_top5','context')
LABELS = ('alignment_or_chunking','candidate_retrieval','fusion_ranking','document_quota','context_budget')


def probe_evidence(chunks, trace):
    did = trace['document_id']
    if not chunks or any(c['paper_id'] != did for c in chunks):
        raise ValueError('Evidence probes require a single document namespace')
    lookup = {c['chunk_id']:c for c in chunks}
    if len(lookup)!=len(chunks): raise ValueError('Duplicate chunk identity')
    stages = trace['stage_ids']
    if set(stages)!=set(STAGES): raise ValueError('Incomplete stage trace')
    previous = set(lookup)
    for name in STAGES:
        ids = stages[name]
        if len(set(ids))!=len(ids) or not set(ids)<=previous:
            raise ValueError('Stages must be nested subsets')
        previous=set(ids)
    units=[]
    for anchor in trace['aligned']:
        coverage={name:evidence_coverage([lookup[c] for c in stages[name]],[anchor]) for name in STAGES}
        label=next((LABELS[i] for i,name in enumerate(STAGES) if coverage[name]['full_evidence_recall']<1),'covered')
        ranks={}
        for branch,ordered in trace['branch_ranks'].items():
            if len({r['chunk_id'] for r in ordered})!=len(ordered): raise ValueError('Duplicate branch hit')
            if any(r['chunk_id'] not in lookup or type(r['rank']) is not int or r['rank']<1 for r in ordered):
                raise ValueError('Invalid branch hit')
            relevant=[r['rank'] for r in ordered if evidence_coverage([lookup[r['chunk_id']]],[anchor])['evidence_recall']>0]
            ranks[branch]=min(relevant) if relevant else None
        fused=trace['fused_ids']
        if len(set(fused))!=len(fused) or set(fused)!=set(stages['candidates']): raise ValueError('Fusion population changed')
        full_rank=next((i for i in range(1,len(fused)+1)
            if evidence_coverage([lookup[c] for c in fused[:i]],[anchor])['full_evidence_recall']==1),None)
        units.append({'evidence_id':anchor['evidence_id'],'alignment_status':anchor['status'],
            'first_loss':label,'coverage':coverage,'first_overlap_branch_rank':ranks,
            'full_coverage_within_paper_rank':full_rank})
    return {'question_id':trace['question_id'],'variant':trace['variant'],'paper_id':trace['paper_id'],
            'document_id':did,'source_question_id':trace['source_question_id'],'units':units,
            'stage_chunk_counts':{name:len(stages[name]) for name in STAGES}}


def summarize_probes(records):
    variants={}
    for key in sorted({r['variant'] for r in records}):
        rows=[u for r in records if r['variant']==key for u in r['units']]
        counts=Counter(u['first_loss'] for u in rows)
        variants[key]={'evidence_unit_occurrences':len(rows),'first_loss_counts':dict(counts)}
    return variants
