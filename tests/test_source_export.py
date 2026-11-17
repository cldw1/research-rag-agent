from dataclasses import replace

from src.chat_service import ResearchChatService
from src.llm.provider import Completion
from src.storage.conversations import source_location_label
from tests.chat_helpers import FakeProvider
from tests.test_multidoc import collection


def test_location_uses_saved_pdf_page_or_real_text_lines_and_word_blocks():
    assert source_location_label({'page_number':3}) == 'PDF 第 3 页'
    assert source_location_label({'source_metadata':{'spans':[{'line_start':3,'line_end':5}]}}) == '原文第 3–5 行'
    word={'section':'Methods','source_metadata':{'spans':[{'block_id':'b000002','paragraph_index':2}]}}
    assert source_location_label(word) == 'Methods · 结构块 b000002'
    assert 'PDF' not in source_location_label(word)


def test_export_labels_each_citation_with_its_filename_and_preserves_literal_evidence(collection):
    r=collection
    llm=FakeProvider(replace(r.config,supports_tools=False),[Completion('Two reports [1, 2].')])
    result=ResearchChatService(r.conversations).ask(r.cid,'Compare',llm,r.retriever,'export')
    text=r.conversations.export_markdown(r.cid)
    assert '[1] paper.md' in text and '[2] second.md' in text
    assert '原文第' in text and 'PDF 第' not in text
    for source in result['cited_sources']:
        saved=next(s for s in r.conversations.sources(r.cid) if s['citation']==source['citation'])
        assert saved['text'] in text
    assert '```text' in text


def test_evidence_markdown_fences_cannot_end_the_verbatim_block():
    from src.storage.conversations import _verbatim_markdown
    source='A literal example\n```python\nprint(1)\n```\n![image](https://example.com/image.png)'
    wrapped=_verbatim_markdown(source)
    assert wrapped.startswith('````text\n') and wrapped.endswith('\n````')
    assert source in wrapped
