"""Fixed-stratum case selection and numeric RRF explanations, without labels changing ranks."""

import re

from src.evaluation.qasper_benchmark import paragraph_ids
from src.retrieval.rank_fusion import fused_order, reciprocal_rank_scores


STRATA = (
    ("fusion_dense_top1", 2), ("hybrid_rank6_10", 2),
    ("outside_both_pools", 1), ("candidate_present_beyond_top10", 1),
)


def select_cases(audit_records, baseline_records):
    baseline = {row["question_id"]: row for row in baseline_records}
    if len(baseline) != len(baseline_records):
        raise ValueError("Duplicate baseline question IDs")
    groups = {name: [] for name, _ in STRATA}
    for record in audit_records:
        row = baseline[record["question_id"]]
        hybrid_at10 = row["methods"]["hybrid"]["10"]["hit"]
        any_at10 = any(row["methods"][name]["10"]["hit"] for name in ("bm25", "dense", "hybrid"))
        outside = record["first_dense_gold_rank"] is None and record["first_bm25_gold_rank"] is None
        if record["category"] == "fusion_regression" and record["first_dense_gold_rank"] == 1:
            groups["fusion_dense_top1"].append(record)
        if record["category"] == "all_miss" and hybrid_at10:
            groups["hybrid_rank6_10"].append(record)
        if outside:
            groups["outside_both_pools"].append(record)
        if record["category"] == "all_miss" and not any_at10 and not outside:
            groups["candidate_present_beyond_top10"].append(record)
    selected = []
    for name, target in STRATA:
        seen_papers = set()
        for record in sorted(groups[name], key=lambda item: (item["question_id"], item["paper_id"])):
            if record["paper_id"] in seen_papers:
                continue
            seen_papers.add(record["paper_id"])
            selected.append({"case": f"C{len(selected)+1:02d}", "stratum": name,
                             "paper_id": record["paper_id"], "question_id": record["question_id"]})
            if len(seen_papers) == target:
                break
        if len(seen_papers) != target:
            raise ValueError("Insufficient distinct papers in selection stratum: " + name)
    return {"rule": "question_id ascending, at most one question per paper within each stratum",
            "cohort_counts": {name: len(items) for name, items in groups.items()}, "cases": selected}


def score_explanation(index, rankings, *, constant=60):
    contributions = {name: 1 / (constant + ranks[index]) if index in ranks else 0.0
                     for name, ranks in rankings.items()}
    present = any(index in ranking for ranking in rankings.values())
    return {"bm25_rank": rankings.get("bm25", {}).get(index),
            "dense_rank": rankings.get("dense", {}).get(index),
            "bm25_contribution": contributions.get("bm25", 0.0),
            "dense_contribution": contributions.get("dense", 0.0),
            "rrf_score": sum(contributions.values()) if present else None,
            "in_candidate_union": present}


def inspect_rankings(chunks, question, rankings, lexical_scores, full_bm25, full_dense):
    order = fused_order(rankings)
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    full_ranks = {"bm25": {int(index): rank for rank, index in enumerate(full_bm25, 1)},
                  "dense": {int(index): rank for rank, index in enumerate(full_dense, 1)}}
    gold = [index for index, chunk in enumerate(chunks) if paragraph_ids(chunk) & question.gold]
    def describe(index):
        return {"chunk_ordinal": int(index), "gold_source_overlap": index in gold,
            "hybrid_rank": ranks.get(index), "full_bm25_rank": full_ranks["bm25"].get(index),
            "full_dense_rank": full_ranks["dense"].get(index), "bm25_score": float(lexical_scores[index]),
            "source_block_ids": sorted(paragraph_ids(chunks[index])),
            **score_explanation(index, rankings)}
    # Numeric explanation is checked against the same unweighted score implementation.
    scores = reciprocal_rank_scores(rankings)
    for index in order:
        if describe(index)["rrf_score"] != scores[index]:
            raise ValueError("Score contribution mismatch")
    complete_order = fused_order(full_ranks)
    complete_rank = {index: rank for rank, index in enumerate(complete_order, 1)}
    return {"gold_paragraph_ids": sorted(question.gold), "gold_chunks": [describe(index) for index in gold],
            "top5": [describe(index) for index in order[:5]], "hybrid_top10_ordinals": order[:10],
            "best_gold_hybrid_rank": min((ranks[index] for index in gold if index in ranks), default=None),
            "counterfactual_full_candidate_pool": {
                "candidate_k": len(chunks),
                "best_gold_hybrid_rank": min((complete_rank[index] for index in gold), default=None),
                "meaning": "Diagnostic recomputation using the same scores and RRF constant, with every chunk as a candidate"}}


def fenced_reference(text):
    """Keep external text inert even when it contains Markdown or code fences."""
    longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return fence + "text\n" + text + "\n" + fence
