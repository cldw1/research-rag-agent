"""Isolated two-process lifecycle acceptance; real BGE, optional live DeepSeek."""

import argparse
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import build_opener, ProxyHandler
from unittest.mock import Mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pdf_answers import read_json, write_new
from src.evaluation.pdf_benchmark import sha256


def fixture_documents():
    """Generated examples, not real-paper experiments or performance evidence."""
    import pymupdf
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 65), 'Synthetic report A: random sampling.', fontsize=14)
        page.insert_text((50, 95), 'The fixed validation sample contains 48 observations.', fontsize=13)
        pdf.set_metadata({'title': 'Acceptance Report', 'author': 'Synthetic fixture'})
        pdf_bytes = pdf.tobytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Synthetic report B: stratified sampling.</w:t></w:r></w:p><w:p><w:r><w:t>The fixed validation sample contains 96 observations.</w:t></w:r></w:p></w:body></w:document>')
        archive.writestr('docProps/core.xml', '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Acceptance Report</dc:title></cp:coreProperties>')
    return [('report_a.pdf', pdf_bytes), ('report_b.docx', buffer.getvalue()),
            ('notes.md', b'# Synthetic notes\n\nAn additional document describes retrieval latency monitoring.'),
            ('outside.txt', b'UNSELECTED_MARKER: This unselected synthetic file must never enter the selected chat.')]


class AcceptanceProvider:
    def __init__(self, config, retriever, folder, *, live=False, proxy=False):
        self.config, self.retriever, self.folder = config, retriever, folder
        self.requests, self.usage = [], []
        self.client, self.delegate = None, None
        if live:
            from dotenv import load_dotenv
            from openai import OpenAI, DefaultHttpxClient
            from src.llm.provider import OpenAICompatibleProvider
            load_dotenv(ROOT/'.env', override=False)
            key = os.getenv('DEEPSEEK_API_KEY', '').strip()
            if not key:
                raise ValueError('Configure the existing DEEPSEEK_API_KEY locally')
            self.client = OpenAI(api_key=key, base_url=config.base_url, max_retries=0, timeout=60.,
                http_client=DefaultHttpxClient(trust_env=proxy, follow_redirects=False))
            self.delegate = OpenAICompatibleProvider(config, key, client=self.client)

    def complete(self, messages, **kwargs):
        from src.llm.provider import Completion, ToolCall
        # Persist before sending; never replay an existing attempt automatically.
        number = len(self.requests)+1
        if number > 5:
            raise RuntimeError('Acceptance request limit exceeded')
        request = {'messages': messages, **kwargs}
        write_new(self.folder/f'{number}-request.json', request)
        self.requests.append(request)
        if self.delegate:
            response = self.delegate.complete(messages, **kwargs)
        elif number == 1:
            response = Completion(tool_calls=(ToolCall('body', 'search_research_paper', json.dumps({
                'paper_id': self.retriever.paper_id, 'question': 'Compare random and stratified sampling validation sample counts'})),))
        else:
            payload = json.loads(messages[-1]['content'])
            numbers = [next(s['citation'] for s in payload['sources'] if s['paper_id']==did)
                       for did in self.retriever.document_ids]
            response = Completion(f'Synthetic report A uses random sampling with 48 observations [{numbers[0]}]; '
                                  f'report B uses stratified sampling with 96 observations [{numbers[1]}].')
        self.usage.append(response.usage)
        write_new(self.folder/f'{number}-response.json', asdict(response))
        return response

    def close(self):
        if self.client:
            self.client.close()


def worker(folder, phase, *, live=False, proxy=False):
    import torch
    from src.chat_service import ResearchChatService
    from src.embedding import load_embedding_resources
    from src.ingestion.parsers import UnifiedDocumentParser
    from src.llm.provider import ModelConfig
    from src.storage.artifacts import StorageError
    from src.storage.backup import create_backup, restore_backup
    from src.storage.conversations import ConversationStore
    from src.storage.document_store import LocalDocumentStore

    folder = Path(folder).resolve()
    write_new(folder/f'{phase}-intent.json', {'phase':phase, 'live':live})
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(8)
    model, counter, profile = load_embedding_resources(local_files_only=True)
    store = LocalDocumentStore(folder/'catalog')
    provider = None
    try:
        chats = ConversationStore(store)
        if phase == 'build':
            documents = []
            builds = []
            fixtures = fixture_documents()
            for name, content in fixtures:
                document = store.ingest(content, name).document
                assert document.status == 'ready'
                assert document.file_type == 'pdf' or all(b.location.page_number is None for b in document.blocks)
                documents.append(document.document_id)
                builds.append(store.build_index(document.document_id, model, profile, counter).index_id)
            assert store.ingest(fixtures[0][1], fixtures[0][0]).reused
            assert store.build_index(documents[0], model, profile, counter).reused
            config = ModelConfig(temperature=0., max_output_tokens=700)
            cid = chats.create(documents[:2], config, 'Synthetic multi-format acceptance')
            state = {'documents':documents, 'indexes':builds, 'conversation_id':cid, 'config':config.to_dict()}
            write_new(folder/'state.json', state)
        else:
            state = read_json(folder/'state.json')
            documents, cid = state['documents'], state['conversation_id']
            config = ModelConfig(**state['config'])
            assert len(chats.history(cid)) == 1
            assert sha256(chats.export_markdown(cid).encode()) == read_json(folder/'build.json')['export_sha256']
            # Restoring the library must not reparse files or encode documents.
            UnifiedDocumentParser.parse = Mock(side_effect=AssertionError('Unexpected reparse'))
        encode = model.encode
        model.encode = Mock(side_effect=AssertionError('Scope restoration must not encode documents'))
        retriever = store.load_scoped_retriever(documents[:2], model, profile)
        model.encode.assert_not_called()
        model.encode = Mock(wraps=encode)
        previous = chats.sources(cid)
        provider = AcceptanceProvider(config, retriever, folder/'api_attempts'/phase, live=live, proxy=proxy)
        question = ('Compare the two selected synthetic reports: state each sampling method and validation sample count. Cite each report separately.'
                    if phase=='build' else 'Which of the two reports used more validation observations? Use our selected reports and cite the evidence.')
        result = ResearchChatService(chats).ask(cid, question, provider, retriever, phase)
        assert not result['invalid_citations'] and not result['truncated'] and result['cited_numbers']
        assert set(result['evidence_document_ids']) == set(documents[:2])
        if phase == 'build':
            assert {s['paper_id'] for s in result['cited_sources']} == set(documents[:2])
        assert 'UNSELECTED_MARKER' not in json.dumps(provider.requests)
        if phase == 'restore':
            old_answer = chats.history(cid)[0]['result']['answer']
            assert any(m.get('role')=='assistant' and m.get('content')==old_answer for m in provider.requests[0]['messages'])
        sources = chats.sources(cid)
        for old in previous:
            assert next(s for s in sources if s['citation']==old['citation']) == old
        count = len(provider.requests)
        assert ResearchChatService(chats).ask(cid, question, provider, retriever, phase) == result
        assert len(provider.requests) == count
        markdown = chats.export_markdown(cid)
        assert 'report_a.pdf' in markdown and 'report_b.docx' in markdown and 'PDF 第 1 页' in markdown
        assert '结构块' in markdown and '```text' in markdown
        export = folder/f'{phase}-conversation.md'
        with export.open('x', encoding='utf-8') as stream:
            stream.write(markdown)
        output = {'phase':phase, 'pid':os.getpid(), 'indexed_formats':['pdf','docx','md','txt'],
            'selected_documents':2, 'history_turns':len(chats.history(cid)), 'scope_bindings':retriever.document_bindings,
            'restored_without_document_encoding':True, 'query_encoding_calls':model.encode.call_count,
            'live_api_calls':len(provider.requests) if live else 0, 'scripted_calls':0 if live else len(provider.requests),
            'returned_usage':provider.usage, 'model_config':config.to_dict(), 'cited_numbers':result['cited_numbers'],
            'source_numbers_stable':True, 'unselected_document_excluded':True, 'request_idempotency':True,
            'export_sha256':sha256(markdown.encode())}
        if phase == 'restore':
            before = chats.history(cid)
            create_backup(store, folder/'backup')
            restore_backup(folder/'backup', folder/'restored')
            recovered = LocalDocumentStore(folder/'restored')
            try:
                copied = ConversationStore(recovered)
                assert copied.history(cid)==before and copied.sources(cid)==sources
                assert copied.export_markdown(cid)==markdown
                assert len(recovered.list_documents())==4
            finally:
                recovered.close()
            originals = {p for p in (store.root/'objects').rglob('*') if p.is_file()}
            store.delete_document(documents[1])
            assert cid not in {c['id'] for c in chats.list()}
            try:
                retriever.search('validation sample', retriever.paper_id)
            except StorageError:
                pass
            else:
                raise AssertionError('Deleted document remained searchable')
            assert {p for p in (store.root/'objects').rglob('*') if p.is_file()} == originals
            store.restore_document(documents[1])
            assert store.build_index(documents[1], model, profile, counter).reused
            rebuilt = store.build_index(documents[1], model, profile, counter, force=True)
            assert rebuilt.index_id != state['indexes'][1]
            try:
                chats.details(cid, require_active=True)
            except StorageError:
                pass
            else:
                raise AssertionError('Old conversation silently switched index versions')
            assert chats.history(cid)==before and chats.export_markdown(cid)==markdown
            temporary = chats.create(documents[2], config, 'Rename me')
            chats.rename(temporary, 'Renamed fixture')
            assert chats.details(temporary)['title']=='Renamed fixture'
            chats.delete(temporary)
            assert temporary not in {c['id'] for c in chats.list()}
            output.update(backup_restore_preserves_export=True, document_removal_invalidates_cached_scope=True,
                          reindex_preserves_history_and_requires_new_chat=True, conversation_rename_delete=True)
        write_new(folder/f'{phase}.json', output)
        print({'phase':phase, 'status':'passed', 'live_api_calls':output['live_api_calls']}, flush=True)
    finally:
        if provider:
            provider.close()
        store.close()


def check_http(folder):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with (folder/'streamlit.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(ROOT/'app.py'),
            '--server.address', '127.0.0.1', '--server.port', str(port), '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'], cwd=ROOT,
            env={**os.environ, 'COPILOT_DATA_DIR':str(folder/'restored')}, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic()+20
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError('Streamlit stopped before health check')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as response:
                        if response.status==200 and response.read()==b'ok':
                            return {'loopback_health':True, 'port':port, 'scope':'HTTP startup only; page rendering covered separately by AppTest'}
                except OSError:
                    pass
                time.sleep(.25)
            raise RuntimeError('Streamlit health check timed out')
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['build','restore'])
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--use-environment-proxy', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.folder, args.worker, live=args.live, proxy=args.use_environment_proxy)
        return
    if args.output and args.output.exists():
        raise FileExistsError('Output exists')
    parent = ROOT/'data/acceptance_checks'
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    names = ['scripts/acceptance_workflow.py', 'src/storage/conversations.py', 'src/chat_service.py',
             'src/agent/research_agent.py', 'src/ingestion/parsers.py', 'src/llm/provider.py']
    source_hashes = {name:sha256((ROOT/name).read_bytes()) for name in names}
    for name in names:
        target = folder/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write_new(folder/'manifest.json', {'source_sha256':source_hashes, 'live':args.live,
                                      'environment_proxy':args.use_environment_proxy, 'max_api_calls':10 if args.live else 0})
    print({'workspace':str(folder), 'generation':'live DeepSeek' if args.live else 'scripted'}, flush=True)
    for phase in ('build','restore'):
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', phase, '--folder', str(folder)]
        if args.live:
            cmd.append('--live')
        if args.use_environment_proxy:
            cmd.append('--use-environment-proxy')
        subprocess.run(cmd, cwd=ROOT, check=True)
    build, restored = read_json(folder/'build.json'), read_json(folder/'restore.json')
    assert build['pid'] != restored['pid'] and build['scope_bindings'] == restored['scope_bindings']
    assert (build['history_turns'], restored['history_turns']) == (1,2)
    assert all(sha256((ROOT/name).read_bytes())==expected for name,expected in source_hashes.items())
    report = {'scope':'Generated four-format lifecycle fixtures, not research answer-quality metrics',
        'generation':'live DeepSeek' if args.live else 'scripted', 'run_id':folder.name,
        'processes':[build,restored], 'http':check_http(folder),
        'total_live_api_calls':build['live_api_calls']+restored['live_api_calls'],
        'source_sha256':source_hashes,
        'ocr':'Not exercised: requires separately prepared local OCR models/language data',
        'browser_interactions':'Not automated; rendering and controls covered separately by Streamlit AppTest'}
    write_new(folder/'summary.json', report)
    if args.output:
        write_new(args.output, report)
    print({'summary':str(folder/'summary.json'), 'total_live_api_calls':report['total_live_api_calls'], 'status':'passed'})


if __name__=='__main__':
    main()
