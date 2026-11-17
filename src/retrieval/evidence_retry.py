"""Bounded, opt-in retrieval after an explicit evidence-insufficiency response.

The gate reads no gold labels. The follow-up query is always the user's original
question, never instructions or search targets extracted from a document/answer.
"""

from dataclasses import dataclass
from collections import deque
from itertools import zip_longest
import re

from src.generation.context_builder import build_qa_prompt
from src.generation.strict_context_builder import build_strict_context


@dataclass(frozen=True)
class EvidenceRetryPolicy:
    enabled: bool = False
    max_chars: int = 7400
    branch_top_k: int = 12
    candidate_k: int = 50
    retained_anchors: int = 2
    max_new_chunks: int = 4
    version: str = "explicit-insufficiency-branch-diversity-v1"
    preserve_citations: bool = False
    balance_novel: bool = False
    per_document_search: bool = False

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in (
            self.max_chars, self.branch_top_k, self.candidate_k,
            self.retained_anchors, self.max_new_chunks,
        )):
            raise ValueError("Retry limits must be positive integers")
        if type(self.per_document_search) is not bool:
            raise ValueError("Per-document search must be a boolean")


INSUFFICIENCY_PATTERNS = (
    r"\binsufficient\b",
    r"\bnot (?:contain |provide )?enough (?:information|detail|evidence)",
    r"\bcannot (?:reliably |fully )?(?:answer|determine|verify|assess|confirm|list)\b",
    r"\bcannot be (?:fully )?answered\b",
    r"\b(?:evidence|passages)\b[^.\n]{0,65}\b(?:does|do) not (?:contain|provide|report)\b",
    r"证据不足|信息不足|无法(?:回答|确定|核实)|未(?:提供|报告).{0,20}(?:信息|数值|数据)",
)


def insufficiency_signal(answer, *, include_passive=False):
    """A coarse explicit-abstention heuristic, not an answer-quality classifier."""
    if not isinstance(answer, str) or not answer.strip():
        return None
    patterns = INSUFFICIENCY_PATTERNS
    if include_passive:
        patterns += (r"\bcannot be (?:fully )?(?:determined|verified|confirmed)\b",)
    for index, pattern in enumerate(patterns):
        if re.search(pattern, answer, re.IGNORECASE):
            return f"explicit_insufficiency_pattern_{index+1}"
    return None


def prepare_evidence_retry(retriever, question, paper_id, answer, baseline_sources, *,
                           policy=None, rounds_used=0, on_search=None):
    policy = policy or EvidenceRetryPolicy()
    decision = {"policy_version": policy.version, "retry": False,
                "retrieval_calls": 0, "new_chunk_ids": [], "dropped_chunk_ids": [],
                "retrieval_mode": "per_document" if policy.per_document_search else "joint",
                "retrieval_document_ids": [], "new_evidence_counts": {}}
    if not policy.enabled:
        return {**decision, "reason": "disabled"}
    if type(rounds_used) is not int or rounds_used < 0:
        raise ValueError("Invalid retry round count")
    if rounds_used >= 1:
        return {**decision, "reason": "round_limit"}
    signal = insufficiency_signal(answer, include_passive=policy.per_document_search)
    if not signal:
        return {**decision, "reason": "no_explicit_insufficiency"}
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A user question is required")

    data = retriever.get_paper_data(paper_id)
    document_ids = list(data.get("document_ids", [str(paper_id)]))
    allowed = set(document_ids)
    per_document = policy.per_document_search and len(document_ids) > 1
    if per_document and (not 2 <= len(document_ids) <= 5 or len(allowed) != len(document_ids)):
        raise ValueError("Retry needs 2–5 distinct selected documents")
    targets = document_ids if per_document else [paper_id]
    decision.update(retrieval_mode="per_document" if per_document else "joint",
                    retrieval_document_ids=list(targets), new_evidence_counts={did: 0 for did in document_ids})
    chunks = data["chunks"]
    lookup = {item["chunk_id"]: item for item in chunks}
    if len(lookup) != len(chunks) or any(str(c["paper_id"]) not in allowed for c in chunks):
        raise ValueError("Retry corpus has invalid document scope or duplicate chunks")

    base, seen = [], set()
    for source in baseline_sources:
        cid = source["chunk_id"]
        if cid not in lookup or str(source["paper_id"]) != str(lookup[cid]["paper_id"]):
            raise ValueError("Baseline evidence is outside the selected document scope")
        if cid not in seen:
            base.append(lookup[cid])
            seen.add(cid)
    if not base:
        raise ValueError("Baseline sources are required")

    branches = []
    for target in targets:
        for method in ("dense", "bm25"):
            if on_search is not None:
                on_search(str(target), method)
            result = retriever.search(query=question, paper_id=target,
                top_k=policy.branch_top_k, candidate_k=policy.candidate_k, method=method)
            for item in result:
                canonical = lookup.get(item["chunk_id"])
                if canonical is None or any(item.get(key) != canonical.get(key) for key in ("paper_id", "text")):
                    raise ValueError("Retry search returned unknown or altered source evidence")
                if per_document and str(item["paper_id"]) != str(target):
                    raise ValueError("Retry search returned evidence from a different target document")
            branches.append(result)
    decision.update(reason=signal, retrieval_calls=2 * len(targets))

    def fits(items):
        context, sources = build_strict_context(items, max_chars=policy.max_chars)
        return bool(context) and [s["chunk_id"] for s in sources] == [c["chunk_id"] for c in items]

    cited = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer)
             for n in re.findall(r"\d+", group)} if policy.preserve_citations else set()
    citation_ids = {s["chunk_id"] for s in baseline_sources if s["citation"] in cited} if cited else set()
    if cited and cited - {s["citation"] for s in baseline_sources}:
        return {**decision, "reason": "invalid_baseline_citations"}
    protected = {c["chunk_id"] for c in base[:policy.retained_anchors]} | citation_ids
    if per_document:
        # Keep at least one existing body source per paper, including uncited
        # papers after the first two anchors. New evidence must not erase them.
        for did in document_ids:
            first = next((c for c in base if str(c["paper_id"]) == str(did)), None)
            if first is not None:
                protected.add(first["chunk_id"])
    selected = [c for c in base if c["chunk_id"] in protected]
    if not fits(selected):
        return {**decision, "reason": "retained_anchors_exceed_budget"}
    selected_ids = {item["chunk_id"] for item in selected}
    if policy.balance_novel or per_document:
        # Remove old/duplicate entries BEFORE alternating: an old hit must not
        # consume a branch's turn and give the other branch extra new slots.
        filtered = []
        for branch in branches:
            branch_seen = set(seen)
            fresh = []
            for item in branch:
                if item["chunk_id"] not in branch_seen:
                    fresh.append(item)
                    branch_seen.add(item["chunk_id"])
            filtered.append(fresh)
        branches = filtered
    def add_novel(item):
        if item is None or len(decision["new_chunk_ids"]) >= policy.max_new_chunks:
            return False
        cid = item["chunk_id"]
        if cid in seen or cid in selected_ids:
            return False
        canonical = lookup[cid]
        if not fits(selected + [canonical]):
            return False
        selected.append(canonical)
        selected_ids.add(cid)
        decision["new_chunk_ids"].append(cid)
        decision["new_evidence_counts"][str(canonical["paper_id"])] += 1
        return True

    if per_document:
        queues = {}
        for index, did in enumerate(document_ids):
            queues[did] = deque(item for pair in zip_longest(*branches[index*2:index*2+2])
                                for item in pair if item is not None)
        # Papers without retained evidence get the first opportunity. Otherwise
        # keep the selected-document order, one fitting new chunk per paper/round.
        covered = {str(c["paper_id"]) for c in selected}
        order = sorted(document_ids, key=lambda did: str(did) in covered)
        while any(queues.values()) and len(decision["new_chunk_ids"]) < policy.max_new_chunks:
            for did in order:
                if len(decision["new_chunk_ids"]) >= policy.max_new_chunks:
                    break
                while queues[did]:
                    if add_novel(queues[did].popleft()):
                        break
    else:
        # Preserve the original two-branch order for single-document and v2 runs.
        for pair in zip_longest(*branches):
            for item in pair:
                add_novel(item)
    if not decision["new_chunk_ids"]:
        return {**decision, "reason": "no_new_evidence_fits"}
    for item in base:
        if item["chunk_id"] not in selected_ids and fits(selected + [item]):
            selected.append(item)
            selected_ids.add(item["chunk_id"])
    context, sources = build_strict_context(selected, max_chars=policy.max_chars)
    return {**decision, "retry": True, "question": question, "paper_id": str(paper_id),
        "context": context, "context_chars": len(context), "sources": sources,
        "prompt": build_qa_prompt(question, context),
        "dropped_chunk_ids": [c["chunk_id"] for c in base if c["chunk_id"] not in selected_ids]}
