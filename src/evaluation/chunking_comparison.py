"""Paired paragraph-source evaluation across two existing chunking strategies."""

from src.evaluation.qasper_benchmark import fully_covered, paper_bootstrap, paragraph_ids


STRATEGIES = ("adaptive", "structure")
SELECTION_RULE = (
    "Recommend structure for follow-up only if its Anchor-First full-paragraph recall "
    "has a paired paper-bootstrap 95% lower bound above zero, and Hybrid Hit@5 and "
    "MRR@5 do not decrease. Otherwise retain adaptive. This experiment never changes "
    "production defaults or the published Test baseline."
)


def verify_source_coverage(document, chunks):
    """Every original block must survive; headings and non-Gold text count too."""
    ranges = {block.block_id: [] for block in document.blocks}
    originals = {block.block_id: block.text for block in document.blocks}
    for chunk in chunks:
        pieces = []
        for span in chunk["source_metadata"]["spans"]:
            key, start, stop = span["block_id"], span["char_start"], span["char_end"]
            if key not in ranges or not 0 <= start < stop <= len(originals[key]):
                raise ValueError("Invalid source span")
            ranges[key].append((start, stop))
            pieces.append(originals[key][start:stop])
        if chunk["text"] != "\n\n".join(pieces):
            raise ValueError("Chunk text does not match original source spans")
    if any(not fully_covered(ranges[key], len(text)) for key, text in originals.items()):
        raise ValueError("Chunking lost source characters")
    return {"blocks": len(ranges), "paragraphs": sum(k.startswith("p") for k in ranges),
            "merged_paragraph_chunks": sum(len(paragraph_ids(c)) > 1 for c in chunks)}


def summarize_comparison(records):
    if not records or len({r["question_id"] for r in records}) != len(records):
        raise ValueError("Non-empty unique question population required")
    results = {}
    for strategy in STRATEGIES:
        rows = [record["strategies"][strategy] for record in records]
        methods = {method: {level: {metric: sum(row["methods"][method][level][metric] for row in rows) / len(rows)
                    for metric in levels[level]} for level in levels}
                   for method, levels in rows[0]["methods"].items()}
        contexts = {variant: {metric: sum(row["contexts"][variant][metric] for row in rows) / len(rows)
                    for metric in values} for variant, values in rows[0]["contexts"].items()}
        results[strategy] = {"methods": methods, "contexts": contexts}
    paired = {"methods_at_5": {}, "contexts": {}}
    for method in ("bm25", "dense", "hybrid"):
        paired["methods_at_5"][method] = {metric: paper_bootstrap(records,
            lambda r, m=method, t=metric: r["strategies"]["structure"]["methods"][m]["5"][t]
                - r["strategies"]["adaptive"]["methods"][m]["5"][t])
            for metric in ("hit", "recall", "mrr", "ndcg", "full_paragraph_recall", "evidence_f1")}
    for variant in ("top5", "anchor_first"):
        paired["contexts"][variant] = {metric: paper_bootstrap(records,
            lambda r, v=variant, t=metric: r["strategies"]["structure"]["contexts"][v][t]
                - r["strategies"]["adaptive"]["contexts"][v][t])
            for metric in ("recall", "full_paragraph_recall", "all_gold_covered", "evidence_f1")}
    adaptive = results["adaptive"]["methods"]["hybrid"]["5"]
    structure = results["structure"]["methods"]["hybrid"]["5"]
    accepted = (paired["contexts"]["anchor_first"]["full_paragraph_recall"]["low"] > 0
                and structure["hit"] >= adaptive["hit"] and structure["mrr"] >= adaptive["mrr"])
    transitions = {"both_hit": 0, "structure_only_hit": 0, "adaptive_only_hit": 0, "both_miss": 0}
    for record in records:
        a = record["strategies"]["adaptive"]["methods"]["hybrid"]["5"]["hit"]
        s = record["strategies"]["structure"]["methods"]["hybrid"]["5"]["hit"]
        key = "both_hit" if a and s else "structure_only_hit" if s else "adaptive_only_hit" if a else "both_miss"
        transitions[key] += 1
    return {"results": results, "paired_structure_minus_adaptive": paired,
            "recommended_for_followup": "structure" if accepted else "adaptive",
            "hybrid_hit_transitions_at_5": transitions}
