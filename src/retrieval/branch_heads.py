"""Optional branch-leader priority within a fixed multi-document anchor budget."""


def select_branch_heads(document_ids, fused_by_document, branch_heads, *, first_branch='bm25', top_k=5):
    ids=tuple(document_ids)
    if (not 2<=len(ids)<=5 or len(set(ids))!=len(ids) or type(top_k) is not int or top_k<1
            or first_branch not in ('bm25','dense') or set(fused_by_document)!=set(ids) or set(branch_heads)!=set(ids)):
        raise ValueError('Invalid document scope or anchor budget')
    branches=(first_branch,'dense' if first_branch=='bm25' else 'bm25')
    queues={};all_ids=set()
    for did in ids:
        candidates=fused_by_document[did]
        lookup={c['chunk_id']:c for c in candidates}
        if (len(lookup)!=len(candidates) or set(lookup)&all_ids
                or any(c['paper_id']!=did for c in candidates)):
            raise ValueError('Candidate source identity mismatch')
        all_ids.update(lookup)
        if set(branch_heads[did])!={'bm25','dense'}:raise ValueError('Unknown retrieval branch')
        queue=[];seen=set()
        for branch in branches:
            cid=branch_heads[did][branch]
            if cid is not None and cid not in lookup:raise ValueError('Branch leader is outside candidate scope')
            if cid is not None and cid not in seen:
                queue.append(lookup[cid]);seen.add(cid)
        queue.extend(c for c in candidates if c['chunk_id'] not in seen)
        queues[did]=queue
    selected=[]
    for position in range(max((len(q) for q in queues.values()),default=0)):
        for did in ids:
            if position<len(queues[did]):
                selected.append(queues[did][position])
                if len(selected)==top_k:return selected
    return selected
