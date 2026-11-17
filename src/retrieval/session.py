"""Per-request retrieval trace; no query text, credentials or shared mutable log."""

from copy import deepcopy


EVIDENCE_ORDERS = {
    'rrf': 'RRF（默认）',
    'bm25_first': '两路首位优先（BM25 先）',
    'dense_first': '两路首位优先（Dense 先）',
}


class RetrievalSession:
    def __init__(self, retriever):
        self._retriever= retriever
        self._order=getattr(retriever,'evidence_order','rrf')
        self._document_ids=tuple(getattr(retriever,'document_ids',(retriever.paper_id,)))
        self._calls=[]

    def __getattr__(self, name):
        return getattr(self._retriever,name)

    def search(self, *args, **kwargs):
        result=self._retriever.search(*args,**kwargs)
        paper_id=kwargs.get('paper_id',args[1] if len(args)>1 else self._retriever.paper_id)
        method=kwargs.get('method',args[4] if len(args)>4 else 'hybrid')
        is_scope=len(self._document_ids)>1 and paper_id==self._retriever.paper_id
        actual=self._order if is_scope and method=='hybrid' else 'rrf' if method=='hybrid' else method
        self._calls.append({'paper_id':str(paper_id),'method':method,'evidence_order':actual,
            'document_ids':list(self._document_ids) if is_scope else [str(paper_id)],
            'top_k':kwargs.get('top_k',args[2] if len(args)>2 else 5),'result_count':len(result)})
        return result

    def snapshot(self):
        return {'multi_document_order':self._order,'selection_version':'branch-head-priority-v1',
                'body_searches':deepcopy(self._calls)}
