"""Application service reused by the chat page and offline integration tests."""

from src.agent.research_agent import ResearchAgent
from src.storage.artifacts import StorageError
from src.llm.control import GenerationCancelled


class ResearchChatService:
    def __init__(self, conversations):
        self.conversations = conversations

    def ask(self, conversation_id, question, provider, retriever, request_id, *, cancellation=None, on_event=None):
        if cancellation:
            cancellation.check()
        details = self.conversations.details(conversation_id, require_active=True)
        if (retriever.paper_id != details["document_id"]
                or getattr(retriever, "index_id", None) != details["index_id"]):
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
            agent = ResearchAgent(provider, retriever, sources=self.conversations.sources(conversation_id))
            result = agent.run(question, history, cancellation=cancellation, on_event=on_event)
            if cancellation:
                cancellation.check()
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
