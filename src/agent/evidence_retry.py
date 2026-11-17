"""One optional follow-up using the agent's stable citations and complete turns."""

from copy import deepcopy
from dataclasses import asdict, replace

from src.agent.research_agent import register_evidence, restore_registry
from src.agent.tool_schema import TOOLS
from src.llm.control import GenerationCancelled
from src.retrieval.evidence_retry import insufficiency_signal, prepare_evidence_retry
from src.generation.citations import extract_citation_numbers


def apply_evidence_retry(agent, result, history, policy):
    if not policy or not policy.enabled:
        return result
    meta = {"enabled": True, "status": "not_triggered", "reason": "no_explicit_insufficiency",
            "policy_version": policy.version, "model_attempts": 0, "new_sources": [],
            "retrieval_calls": 0, "retrieval_document_ids": [], "new_evidence_counts": {},
            "retrieval_mode": "per_document" if policy.per_document_search and len(agent.document_ids)>1 else "joint"}
    def finish(status, reason):
        meta.update(status=status, reason=reason)
        agent._notify("retry_end", status=status, text=result["answer"])
        return {**result, "evidence_retry": meta}
    if not insufficiency_signal(result["answer"], include_passive=policy.per_document_search):
        return {**result, "evidence_retry": meta}
    if result.get("truncated"):
        return finish("skipped", "initial_answer_truncated")
    agent._notify("retry_start")
    sources = [s for s in result["registered_sources"]
               if (s["paper_id"], s["chunk_id"]) in agent.sent_this_turn]
    registry = agent.registry
    sent = set(agent.sent_this_turn)
    usage, contexts = list(result["request_usage"]), list(result["context_selections"])
    def record_search(target, method):
        if agent.cancellation:
            agent.cancellation.check()
        meta["retrieval_calls"] += 1
        if target not in meta["retrieval_document_ids"]:
            meta["retrieval_document_ids"].append(target)
    try:
        if agent.cancellation:
            agent.cancellation.check()
        decision = prepare_evidence_retry(agent.retriever, result["question"], agent.paper_id,
            result["answer"], sources, policy=replace(policy, max_chars=min(policy.max_chars, agent.search.pipeline.max_chars)),
            on_search=record_search)
        meta.update(retrieval_calls=decision["retrieval_calls"],
                    dropped_chunk_ids=decision["dropped_chunk_ids"],
                    retrieval_mode=decision["retrieval_mode"],
                    retrieval_document_ids=decision["retrieval_document_ids"],
                    new_evidence_counts=decision["new_evidence_counts"])
        if not decision["retry"]:
            return finish("skipped", decision["reason"])
        trial_registry = restore_registry(result["registered_sources"])
        payload = register_evidence({"evidence": decision["context"], "sources": decision["sources"]},
                                    trial_registry, set())
        # Retain paired tool exchanges. Add evidence only as user reference data,
        # never a system instruction; the draft answer stays out of this request.
        current = deepcopy(result["messages"][:-1])
        current[0]["reference_context"] = payload["evidence"]
        keep_definitions = getattr(agent.provider, "requires_tool_definitions_for_history", False)
        tools = TOOLS if keep_definitions and any(m.get("tool_calls") for m in current) else None
        selection = agent.window.select(agent.system, history, current, tools)
        contexts.append({**asdict(selection), "messages": None})
        meta["model_attempts"] = 1
        response = agent._request(selection.messages, tools=tools, tool_choice="none" if tools else None)
        usage.append(response.usage)
        answer = response.content.strip()
        numbers = set(extract_citation_numbers(answer))
        supplied = {s["citation"] for s in sources + payload["sources"]}
        if response.tool_calls or not answer or response.finish_reason == "length" or numbers - supplied:
            result = {**result, "request_usage": usage, "model_calls": len(usage), "context_selections": contexts}
            return finish("fallback", "invalid_retry_response")
        agent.registry = trial_registry
        agent.sent_this_turn.update((s["paper_id"], s["chunk_id"]) for s in payload["sources"])
        updated = agent._finish(result["question"], current, response,
            result["tool_calls"] + [{"tool_name": "evidence_retry", "status": "completed"}],
            contexts, usage, rag_context=payload)
        # The common finalizer also understands grouped citation numbers here.
        updated["cited_numbers"] = sorted(numbers)
        updated["cited_sources"] = [agent.registry.get_source(n) for n in sorted(numbers)]
        updated["invalid_citations"] = []
        updated["missing_citations"] = not bool(numbers)
        meta.update(status="accepted", reason=decision["reason"], initial_answer=result["answer"],
            new_sources=[s for s in payload["sources"] if s["chunk_id"] in decision["new_chunk_ids"]])
        agent._notify("retry_end", status="accepted", text=answer)
        return {**updated, "evidence_retry": meta}
    except GenerationCancelled:
        raise
    except Exception:
        # Do not expose raw SDK errors, credentials or external document text.
        if agent.cancellation:
            agent.cancellation.check()
        if meta["model_attempts"] and len(usage) == len(result["request_usage"]):
            usage.append({})  # attempted response usage unknown, not zero tokens
        result = {**result, "request_usage": usage, "model_calls": len(usage), "context_selections": contexts}
        return finish("fallback", "retry_failed")
    finally:
        if meta["status"] != "accepted":
            agent.registry, agent.sent_this_turn = registry, sent
