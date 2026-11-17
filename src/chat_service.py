"""Application service reused by the chat page and offline integration tests."""

from src.agent.research_agent import ResearchAgent
from src.storage.artifacts import StorageError
from src.llm.control import GenerationCancelled
from src.retrieval.session import RetrievalSession
from src.generation.answer_policy import policy_version


class ResearchChatService:
    def __init__(self, conversations):
        self.conversations = conversations

    def ask(self, conversation_id, question, provider, retriever, request_id, *, cancellation=None, on_event=None,
            retry_policy=None, answer_policy='legacy'):
        policy_version(answer_policy)  # Reject invalid choices before creating a turn.
        if cancellation:
            cancellation.check()
        details = self.conversations.details(conversation_id, require_active=True)
        expected = {item["document_id"]: item["index_id"] for item in details["documents"]}
        actual = getattr(retriever, "document_bindings", None)
        if actual is None:
            actual = {retriever.paper_id: getattr(retriever, "index_id", None)}
        if actual != expected:
            raise StorageError("retriever_scope_mismatch", "检索器与会话的文档或索引版本不一致")
        turn_id, existing = self.conversations.begin(conversation_id, question, provider.config, request_id)
        if existing is not None:
            return existing
        cancel_key = cancellation.register(lambda: self.conversations.cancel(conversation_id, turn_id)) if cancellation else None
        try:
            history = []
            for turn in self.conversations.history(conversation_id):
                if turn["status"] != "completed":
                    continue
                old = turn["model_config"]
                if any(old.get(key) != getattr(provider.config, key) for key in ("provider", "model", "base_url")):
                    # Keep foreign native signatures/tool exchanges in storage;
                    # send only portable user/answer text to the new model.
                    messages = [{"role": "user", "content": turn["messages"][0]["content"]},
                                {"role": "assistant", "content": turn["messages"][-1]["content"]}]
                else:
                    messages = turn["messages"]
                history.append(messages)
            retrieval_session=RetrievalSession(retriever)
            agent = ResearchAgent(provider, retrieval_session, sources=self.conversations.sources(conversation_id),
                                  answer_policy=answer_policy)
            result = agent.run(question, history, cancellation=cancellation, on_event=on_event)
            if retry_policy and retry_policy.enabled:
                from src.agent.evidence_retry import apply_evidence_retry
                result = apply_evidence_retry(agent, result, history, retry_policy)
            if cancellation:
                cancellation.check()
            result['retrieval_settings']=retrieval_session.snapshot()
            self.conversations.complete(conversation_id, turn_id, result)
            return result
        except Exception as exc:
            if cancellation and cancellation.event.is_set():
                self.conversations.cancel(conversation_id, turn_id)
                raise GenerationCancelled() from None
            self.conversations.fail(conversation_id, turn_id, getattr(exc, "code", "generation_failed"))
            raise
        finally:
            if cancellation:
                cancellation.unregister(cancel_key)
