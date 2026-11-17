"""Paper-scoped evidence metrics and reviewer-labelled citation ownership checks."""

from src.evaluation.pdf_benchmark import evidence_coverage


def resolve_cases(spec, dataset):
    if spec.get("schema_version") != "multidoc-questions-v1":
        raise ValueError("Unsupported multi-document question schema")
    if dataset.get("status") != "development_pilot" or any(p["split"] != "dev" for p in dataset["papers"]):
        raise ValueError("Development sources only")
    lookup = {q["question_id"]: q for q in dataset["questions"]}
    seen, output = set(), []
    for case in spec["questions"]:
        qid = case["question_id"]
        if qid in seen or not case["question"].strip():
            raise ValueError("Duplicate/empty comparison question")
        seen.add(qid)
        targets = [lookup[q] for q in case["source_question_ids"]]
        if not 2 <= len(targets) <= 5 or len({q["paper_id"] for q in targets}) != len(targets):
            raise ValueError("Each comparison needs different selected papers")
        output.append({**case, "targets": targets})
    return output


def score_context(sources, chunks, targets, bindings, alignments):
    lookup = {c["chunk_id"]: c for c in chunks}
    chosen = []
    for source in sources:
        chunk = lookup[source["chunk_id"]]
        if source["paper_id"] != chunk["paper_id"] or chunk["paper_id"] not in bindings.values():
            raise ValueError("Source document scope mismatch")
        chosen.append(chunk)
    per_paper = {}
    for target in targets:
        if not target["answerable"]:
            continue  # No gold: never turn an absent answer into a retrieval miss.
        did = bindings[target["paper_id"]]
        # Block IDs and page numbers are local to each PDF, not globally unique.
        scoped = [c for c in chosen if c["paper_id"] == did]
        per_paper[target["paper_id"]] = evidence_coverage(scoped, alignments[target["question_id"]])
    positives = list(per_paper.values())
    return {"per_paper_gold": per_paper,
        "document_presence": sum(any(c["paper_id"] == bindings[t["paper_id"]] for c in chosen) for t in targets)/len(targets),
        "mean_full_evidence_recall": sum(p["full_evidence_recall"] for p in positives)/len(positives) if positives else None,
        "all_required_papers_hit": float(all(p["evidence_recall"] > 0 for p in positives)) if positives else None,
        "all_gold_complete": float(all(p["all_evidence_covered"] for p in positives)) if positives else None}


def audit_attribution(claims, sources, allowed_document_ids):
    """Reviewer supplies claim-to-paper labels; this does NOT judge factual support."""
    lookup = {s["citation"]: s for s in sources}
    if len(lookup) != len(sources):
        raise ValueError("Duplicate citation numbers")
    allowed = set(allowed_document_ids)
    checked, seen = [], set()
    for claim in claims:
        if claim["claim_id"] in seen:
            raise ValueError("Duplicate claim")
        seen.add(claim["claim_id"])
        expected, numbers = set(claim["expected_document_ids"]), set(claim["citations"])
        if not expected or not expected <= allowed:
            raise ValueError("Reviewer claim scope is invalid")
        if not numbers:
            status = "missing_citation"
        elif not numbers <= set(lookup):
            status = "unknown_citation"
        else:
            actual = {lookup[n]["paper_id"] for n in numbers}
            status = "out_of_scope" if not actual <= allowed else "matched" if actual == expected else "wrong_document"
        checked.append({"claim_id": claim["claim_id"], "status": status})
    return {"claims": checked, "semantic_faithfulness": None,
            "note": "Ownership consistency for explicitly labelled claims; not semantic fact verification."}
