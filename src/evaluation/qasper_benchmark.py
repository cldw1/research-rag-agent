"""QASPER text-evidence protocol, independent of retrieval and generation."""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.ingestion.document import DocumentBlock, SourceLocation, StructuredDocument


PROTOCOL_VERSION = "qasper-text-retrieval-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


@dataclass(frozen=True)
class EvidenceQuestion:
    paper_id: str
    question_id: str
    question: str
    gold: frozenset[str]
    references: tuple[frozenset[str], ...]


def structured_paper(paper, split):
    """Only title/full_text enter the document; questions/answers never do."""
    blocks, ordinal = [], 0
    for section, (name, paragraphs) in enumerate(zip(
            paper["full_text"]["section_name"], paper["full_text"]["paragraphs"], strict=True)):
        path = (str(name).strip(),) if str(name).strip() else ()
        if path:
            blocks.append(DocumentBlock(f"h{section:04d}", "heading", path[0], path,
                                        SourceLocation(paragraph_index=ordinal)))
            ordinal += 1
        for paragraph, text in enumerate(paragraphs):
            if not text.strip():
                continue
            blocks.append(DocumentBlock(f"p{section:04d}_{paragraph:04d}", "paragraph", text.strip(), path,
                                        SourceLocation(paragraph_index=ordinal)))
            ordinal += 1
    content = {"id": str(paper["id"]), "title": paper["title"], "full_text": paper["full_text"]}
    version = digest({"source": content, "split": split, "converter": PROTOCOL_VERSION})
    return StructuredDocument(document_id=str(paper["id"]), filename=str(paper["id"]) + ".json",
        file_type="qasper", file_hash=digest(content), file_size=len(json.dumps(content).encode()),
        title=paper["title"], author="", parser_version=PROTOCOL_VERSION,
        document_version=version, blocks=tuple(blocks), source_metadata={"dataset": "allenai/qasper", "split": split})


def prepare_dataset(rows, split):
    documents, questions, excluded, counts = [], [], [], Counter()
    seen_papers, seen_questions = set(), set()
    for paper in rows:
        document = structured_paper(paper, split)
        if document.document_id in seen_papers:
            raise ValueError("Duplicate paper ID")
        seen_papers.add(document.document_id)
        documents.append(document)
        lookup = defaultdict(list)
        for block in document.blocks:
            if block.kind == "paragraph":
                lookup[normalize(block.text)].append(block.block_id)
        qas = paper["qas"]
        for qid, text, answers in zip(qas["question_id"], qas["question"], qas["answers"], strict=True):
            qid = str(qid)
            if qid in seen_questions:
                raise ValueError("Duplicate question ID")
            seen_questions.add(qid)
            counts["total_questions"] += 1
            answerable = [a for a in answers["answer"] if not a.get("unanswerable", False)]
            evidence_sets = []
            for answer in answerable:
                evidence = {normalize(t) for t in answer.get("evidence", [])
                            if t.strip() and not t.startswith("FLOAT SELECTED")}
                if evidence:
                    evidence_sets.append(evidence)
            all_evidence = set().union(*evidence_sets) if evidence_sets else set()
            if not answerable:
                reason = "unanswerable"
            elif not all_evidence:
                reason = "no_text_evidence"
            elif any(item not in lookup for item in all_evidence):
                reason = "unmapped_text_evidence"
            elif any(len(lookup[item]) != 1 for item in all_evidence):
                reason = "ambiguous_text_evidence"
            elif not text.strip():
                reason = "empty_question"
            else:
                references = tuple(frozenset(lookup[item][0] for item in evidence) for evidence in evidence_sets)
                questions.append(EvidenceQuestion(document.document_id, qid, text.strip(),
                                                 frozenset().union(*references), references))
                counts["eligible_questions"] += 1
                continue
            counts[reason] += 1
            excluded.append({"paper_id": document.document_id, "question_id": qid, "reason": reason})
    counts["total_papers"] = len(documents)
    counts["eligible_papers"] = len({q.paper_id for q in questions})
    return documents, questions, dict(counts), excluded


def paragraph_ids(chunk):
    return {span["block_id"] for span in chunk["source_metadata"]["spans"]
            if span["block_id"].startswith("p")}


def fully_covered(spans, length):
    end = 0
    for start, stop in sorted(spans):
        if start > end:
            return False
        end = max(end, stop)
    return end >= length


def evidence_coverage(chunks, question, lengths):
    predicted, ranges = set(), defaultdict(list)
    for chunk in chunks:
        predicted.update(paragraph_ids(chunk))
        for span in chunk["source_metadata"]["spans"]:
            if span["block_id"] in question.gold:
                ranges[span["block_id"]].append((span["char_start"], span["char_end"]))
    matched = len(predicted & question.gold)
    complete = sum(fully_covered(ranges[key], lengths[key]) for key in question.gold)
    f1 = max((2 * len(predicted & reference) / (len(predicted) + len(reference))
              for reference in question.references), default=0.0)
    return {"recall": matched / len(question.gold),
            "full_paragraph_recall": complete / len(question.gold),
            "all_gold_covered": float(complete == len(question.gold)), "evidence_f1": f1}


def retrieval_metrics(all_chunks, ranked, question, lengths, k):
    if k < 1 or not question.gold:
        raise ValueError("Non-empty gold and positive K required")
    chosen = ranked[:k]
    flags = [bool(paragraph_ids(chunk) & question.gold) for chunk in chosen]
    relevant = sum(bool(paragraph_ids(chunk) & question.gold) for chunk in all_chunks)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, relevant)))
    dcg = sum(int(flag) / math.log2(rank + 2) for rank, flag in enumerate(flags))
    first = next((rank for rank, flag in enumerate(flags, 1) if flag), None)
    return {"hit": float(any(flags)), "precision": sum(flags) / k,
            "mrr": 1 / first if first else 0.0, "ndcg": dcg / ideal if ideal else 0.0,
            **evidence_coverage(chosen, question, lengths)}


def percentile_summary(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite non-empty observations required")
    return {"n": len(values), "mean": float(values.mean()),
            "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95))}


def paper_bootstrap(records, getter, *, seed=42, resamples=2000):
    """Question-macro means, resampling complete paper clusters with replacement."""
    import numpy as np
    groups = defaultdict(list)
    for record in records:
        groups[record["paper_id"]].append(float(getter(record)))
    if not groups or resamples < 1:
        raise ValueError("Records and resamples required")
    values = [groups[key] for key in sorted(groups)]
    totals = np.asarray([sum(group) for group in values])
    counts = np.asarray([len(group) for group in values])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(resamples, len(values)))
    means = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {"mean": float(totals.sum() / counts.sum()), "low": float(np.quantile(means, .025)),
            "high": float(np.quantile(means, .975)), "confidence": .95,
            "resamples": resamples, "seed": seed, "sampling_unit": "paper"}
