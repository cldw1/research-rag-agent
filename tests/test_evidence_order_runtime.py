from unittest.mock import Mock

import pytest

from tests.test_multidoc import collection,provider
from src.chat_service import ResearchChatService
from src.retrieval.branch_heads import select_branch_heads
from src.retrieval.session import RetrievalSession
from src.storage.artifacts import StorageError
from src.retrieval.evidence_retry import EvidenceRetryPolicy


@pytest.mark.parametrize('mode',['bm25_first','dense_first'])
def test_runtime_selection_matches_experiment_and_encodes_only_one_query(collection,mode):
    r=collection
    r.model.encode=Mock(wraps=r.model.encode)
    runtime=r.store.load_scoped_retriever(r.ids,r.model,r.profile,evidence_order=mode)
    r.model.encode.assert_not_called()
    found=runtime.search('momentum experiment',runtime.paper_id,top_k=5)
    assert r.model.encode.call_count==1
    baseline=r.retriever.search('momentum experiment',r.retriever.paper_id,top_k=len(r.retriever.chunks))
    fused={did:sorted([c for c in baseline if c['paper_id']==did],key=lambda c:c['global_fusion_rank']) for did in r.ids}
    heads={did:{} for did in r.ids}
    for did in r.ids:
        for branch in ('bm25','dense'):
            owned=[c for c in fused[did] if c[branch+'_rank'] is not None]
            heads[did][branch]=min(owned,key=lambda c:c[branch+'_rank'])['chunk_id'] if owned else None
    expected=select_branch_heads(r.ids,fused,heads,first_branch='bm25' if mode=='bm25_first' else 'dense')
    assert [c['chunk_id'] for c in found]==[c['chunk_id'] for c in expected]
    assert all(c['selection']==mode for c in found)
    assert [c['chunk_id'] for c in runtime.search('momentum experiment',runtime.paper_id,top_k=10)[:5]]==[c['chunk_id'] for c in found]
    with pytest.raises(AttributeError):runtime.evidence_order='rrf'


def test_single_document_and_branch_only_calls_keep_their_actual_policy(collection):
    r=collection
    engine=r.store.load_scoped_retriever(r.ids,r.model,r.profile,evidence_order='bm25_first')
    trace=RetrievalSession(engine)
    trace.search('momentum',r.ids[0])
    trace.search('momentum',engine.paper_id,method='dense')
    trace.search('momentum',engine.paper_id)
    snapshot=trace.snapshot()
    assert [c['evidence_order'] for c in snapshot['body_searches']]==['rrf','dense','bm25_first']
    snapshot['body_searches'].clear()
    assert len(trace.snapshot()['body_searches'])==3
    with pytest.raises(ValueError):trace.search('question',r.outside.document_id)
    assert len(trace.snapshot()['body_searches'])==3


def test_saved_turn_settings_survive_switch_reopen_and_idempotent_requests(collection):
    r=collection;service=ResearchChatService(r.conversations)
    first=r.store.load_scoped_retriever(r.ids,r.model,r.profile,evidence_order='bm25_first')
    result=service.ask(r.cid,'compare',provider(r),first,'first')
    assert result['retrieval_settings']['multi_document_order']=='bm25_first'
    assert result['retrieval_settings']['body_searches'][0]['evidence_order']=='bm25_first'
    other=r.store.load_scoped_retriever(r.ids,r.model,r.profile,evidence_order='dense_first')
    unused=provider(r)
    assert service.ask(r.cid,'compare',unused,other,'first')==result and not unused.requests
    second=service.ask(r.cid,'compare again',provider(r,'second'),other,'second')
    assert second['retrieval_settings']['multi_document_order']=='dense_first'
    assert len(second['retrieval_settings']['body_searches'])==1
    from src.storage.document_store import LocalDocumentStore
    from src.storage.conversations import ConversationStore
    reopened=LocalDocumentStore(r.store.root)
    try:
        history=ConversationStore(reopened).history(r.cid)
        assert [t['result']['retrieval_settings']['multi_document_order'] for t in history]==['bm25_first','dense_first']
        exported=ConversationStore(reopened).export_markdown(r.cid)
        assert 'BM25 先' in exported and 'Dense 先' in exported
    finally:reopened.close()


def test_retry_branch_searches_are_audited_separately_from_head_priority(collection):
    r=collection;engine=r.store.load_scoped_retriever(r.ids,r.model,r.profile,evidence_order='bm25_first')
    llm=provider(r)
    from src.llm.provider import Completion
    llm.responses[-1]=Completion('Insufficient evidence [1][2].')
    result=ResearchChatService(r.conversations).ask(r.cid,'compare',llm,engine,'retry',
        retry_policy=EvidenceRetryPolicy(enabled=True,preserve_citations=True,balance_novel=True))
    calls=result['retrieval_settings']['body_searches']
    assert calls[0]['evidence_order']=='bm25_first'
    assert [c['evidence_order'] for c in calls[1:]]==['dense','bm25']
    assert result['evidence_retry']['enabled']


@pytest.mark.parametrize('mode',['read_file','unknown'])
def test_invalid_order_never_creates_a_retriever(collection,mode):
    r=collection
    with pytest.raises(StorageError,match='证据顺序'):
        r.store.load_scoped_retriever(r.ids,r.model,r.profile,evidence_order=mode)


def test_single_document_rejects_an_inapplicable_multi_document_order(collection):
    r=collection
    with pytest.raises(StorageError,match='仅用于多文档'):
        r.store.load_scoped_retriever(r.ids[:1],r.model,r.profile,evidence_order='bm25_first')
