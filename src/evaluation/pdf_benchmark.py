"""PDF-native evidence labels, independent of chunk boundaries and retrieval ranks."""

import hashlib
import inspect
import json
import math
from pathlib import Path
from collections import Counter

from src.evaluation.qasper_benchmark import fully_covered


SCHEMA_VERSION = "pdf-qa-v1"
CATEGORIES = {"fact", "method", "experiment", "limitation", "multi_section",
              "source_conflict", "unanswerable"}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def native_blocks(path):
    """Original PDF page/block coordinates; never read questions or chunk text."""
    import pymupdf

    options = {"sort": True}
    if "use_ocr" in inspect.signature(pymupdf.utils.get_text).parameters:
        options["use_ocr"] = False
    blocks = {}
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("Encrypted benchmark PDF")
        for page_number, page in enumerate(pdf, 1):
            for position, block in enumerate(page.get_text("blocks", **options)):
                if len(block) >= 7 and block[6] == 0 and str(block[4]).strip():
                    blocks[(page_number, position)] = {
                        "text": str(block[4]).strip(),
                        "bbox": [float(v) for v in block[:4]],
                    }
        return blocks, len(pdf), "pymupdf-" + pymupdf.__version__


def validate_dataset(dataset):
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported PDF benchmark schema")
    papers = {p["paper_id"]: p for p in dataset["papers"]}
    if len(papers) != len(dataset["papers"]) or not papers:
        raise ValueError("Duplicate/empty paper manifest")
    fingerprints, arxiv_ids = set(), set()
    for paper in papers.values():
        if paper["split"] not in {"dev", "test"}:
            raise ValueError("Invalid paper split")
        if Path(paper["filename"]).name != paper["filename"] or not paper["filename"].endswith(".pdf"):
            raise ValueError("Expected a PDF basename")
        if len(paper["pdf_sha256"]) != 64 or paper["page_count"] < 1:
            raise ValueError("Invalid source identity")
        if paper["pdf_sha256"] in fingerprints:
            raise ValueError("Duplicate PDF content across paper IDs or splits")
        fingerprints.add(paper["pdf_sha256"])
        if paper.get("arxiv_version"):
            identity = paper["arxiv_version"].split("v", 1)[0]
            if identity in arxiv_ids:
                raise ValueError("Different versions of one arXiv paper cannot occupy multiple splits")
            arxiv_ids.add(identity)
    questions, evidence = set(), set()
    for question in dataset["questions"]:
        qid = question["question_id"]
        if qid in questions or question["paper_id"] not in papers:
            raise ValueError("Duplicate question or unknown paper")
        questions.add(qid)
        if not question["question"].strip() or question["category"] not in CATEGORIES:
            raise ValueError("Invalid question/category")
        if question["annotation"]["human_review"] not in {"pending", "verified"}:
            raise ValueError("Invalid annotation review status")
        if not isinstance(question["answerable"], bool):
            raise ValueError("answerable must be boolean")
        if not question["reference_points"]:
            raise ValueError("Reference answer/rubric required")
        if question["answerable"] != bool(question["evidence"]):
            raise ValueError("Answerable questions need evidence; unanswerable questions have none")
        if (question["category"] == "unanswerable") == question["answerable"]:
            raise ValueError("Unanswerable category mismatch")
        if not question["answerable"] and not question.get("absence_review"):
            raise ValueError("Unanswerable questions need a documented absence review")
        for anchor in question["evidence"]:
            identity = (qid, anchor["evidence_id"])
            if identity in evidence:
                raise ValueError("Duplicate evidence ID within question")
            evidence.add(identity)
            if (not 1 <= anchor["page_number"] <= papers[question["paper_id"]]["page_count"]
                    or anchor["block_index"] < 0 or not 0 <= anchor["char_start"] < anchor["char_end"]
                    or len(anchor["bbox"]) != 4 or not anchor["section"].strip()
                    or len(anchor["text_sha256"]) != 64):
                raise ValueError("Invalid evidence locator")
    return papers


def load_dataset(path):
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(dataset)
    return dataset


def verify_candidate_release(folder):
    """Check frozen selection and exact derived views without scoring any paper."""
    folder = Path(folder)
    manifest = json.loads((folder / "candidate_manifest.json").read_text(encoding="utf-8"))
    names = {"selection.json", "corpus.json", "development.json", "test_candidates.json", "development_protocol.json"}
    if (manifest.get("manifest_version") != "pdf-qa-candidates-v1"
            or set(manifest["artifacts_sha256"]) != names or manifest["final_test_ready"] is not False):
        raise ValueError("Expected unscored PDF candidate release")
    for name in names:
        if sha256((folder / name).read_bytes()) != manifest["artifacts_sha256"][name]:
            raise ValueError("Candidate release checksum mismatch: " + name)
    dataset = load_dataset(folder / "corpus.json")
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    if (dataset["selection_sha256"] != sha256((folder / "selection.json").read_bytes())
            or dataset["status"] != "annotation_candidate"):
        raise ValueError("Paper selection changed after annotation")
    selected = {p["paper_id"]: p for p in selection["papers"]}
    if len(selected) != len(selection["papers"]) or set(selected) != {p["paper_id"] for p in dataset["papers"]}:
        raise ValueError("Paper selection population changed")
    for paper in dataset["papers"]:
        if any(paper[key] != selected[paper["paper_id"]][key] for key in ("title", "arxiv_version", "split", "source_url")):
            raise ValueError("Paper identity or split changed")
    counts = Counter(q["paper_id"] for q in dataset["questions"])
    if (len(dataset["papers"]) != 10 or len(dataset["questions"]) != 150
            or set(counts.values()) != {15} or set(counts) != set(selected)):
        raise ValueError("Expected 10 papers with 15 questions each")
    normalized = [" ".join(q["question"].lower().split()) for q in dataset["questions"]]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate question text")
    if (manifest["paper_count"] != 10 or manifest["question_count"] != 150
            or manifest["answerable"] != sum(q["answerable"] for q in dataset["questions"])
            or manifest["human_review_pending"] != sum(q["annotation"]["human_review"] == "pending" for q in dataset["questions"])
            or manifest["evidence_locators"] != sum(len(q["evidence"]) for q in dataset["questions"])):
        raise ValueError("Candidate release counts changed")
    for split, name, paper_count in (("dev", "development.json", 3), ("test", "test_candidates.json", 7)):
        subset = load_dataset(folder / name)
        papers = [p for p in dataset["papers"] if p["split"] == split]
        ids = {p["paper_id"] for p in papers}
        questions = [q for q in dataset["questions"] if q["paper_id"] in ids]
        if (subset["papers"] != papers or subset["questions"] != questions or len(papers) != paper_count
                or manifest["splits"][split] != {"papers": paper_count, "questions": len(questions)}
                or subset["status"] != ("development_pilot" if split == "dev" else "annotation_candidate")):
            raise ValueError("Derived split differs from authoritative corpus")
    pilot = load_dataset(folder / "pilot.json")
    by_id = {q["question_id"]: q for q in dataset["questions"]}
    if any(by_id.get(q["question_id"]) != q for q in pilot["questions"]):
        raise ValueError("Existing pilot labels changed")
    return {"papers": 10, "questions": 150, "pages": sum(p["page_count"] for p in dataset["papers"]),
            "splits": manifest["splits"], "answerable": manifest["answerable"],
            "evidence_locators": manifest["evidence_locators"],
            "human_review_pending": manifest["human_review_pending"], "final_test_ready": False}


def verify_sources(dataset, source_root):
    """A changed file, page, block or character slice fails closed before ranking."""
    papers = validate_dataset(dataset)
    sources, report = {}, []
    for key, paper in papers.items():
        path = Path(source_root) / paper["filename"]
        if sha256(path.read_bytes()) != paper["pdf_sha256"]:
            raise ValueError(f"PDF checksum mismatch: {key}")
        blocks, pages, engine = native_blocks(path)
        if pages != paper["page_count"] or engine != paper["locator_engine"]:
            raise ValueError(f"PDF page count or locator engine changed: {key}")
        sources[key] = blocks
        report.append({"paper_id": key, "pages": pages, "blocks": len(blocks), "engine": engine})
    count = 0
    for question in dataset["questions"]:
        for anchor in question["evidence"]:
            block = sources[question["paper_id"]].get((anchor["page_number"], anchor["block_index"]))
            if block is None or any(abs(a-b) > .01 for a, b in zip(block["bbox"], anchor["bbox"])):
                raise ValueError(f"Evidence block/bbox mismatch: {question['question_id']}")
            text = block["text"][anchor["char_start"]:anchor["char_end"]]
            if anchor["char_end"] > len(block["text"]) or sha256(text.encode("utf-8")) != anchor["text_sha256"]:
                raise ValueError(f"Evidence text mismatch: {question['question_id']}")
            count += 1
    return sources, {"papers": report, "questions": len(dataset["questions"]),
                     "verified_evidence_locators": count}


def align_evidence(question, native, document):
    """Exact, page-constrained matching; missing/ambiguous evidence remains a miss."""
    aligned = []
    for anchor in question["evidence"]:
        block = native[(anchor["page_number"], anchor["block_index"])]
        gold = block["text"][anchor["char_start"]:anchor["char_end"]]
        matches = []
        for parsed in document.blocks:
            if parsed.location.page_number != anchor["page_number"]:
                continue
            # Bboxes distinguish repeated headers and identical text on the same page.
            box = parsed.location.bbox
            target = anchor["bbox"]
            if box and (min(box[2], target[2]) <= max(box[0], target[0])
                        or min(box[3], target[3]) <= max(box[1], target[1])):
                continue
            start = parsed.text.find(gold)
            while start != -1:
                matches.append((parsed.block_id, start, start + len(gold)))
                start = parsed.text.find(gold, start + 1)
        aligned.append({"evidence_id": anchor["evidence_id"],
                        "status": "mapped" if len(matches) == 1 else "missing" if not matches else "ambiguous",
                        "span": matches[0] if len(matches) == 1 else None})
    return aligned


def evidence_coverage(chunks, aligned):
    if not aligned:
        raise ValueError("Retrieval metrics are undefined for unanswerable questions")
    any_count, full_count = 0, 0
    for anchor in aligned:
        ranges = []
        if anchor["span"] is not None:
            block, start, end = anchor["span"]
            for chunk in chunks:
                for span in chunk["source_metadata"]["spans"]:
                    if span["block_id"] == block:
                        lo, hi = max(start, span["char_start"]), min(end, span["char_end"])
                        if lo < hi:
                            ranges.append((lo-start, hi-start))
            any_count += bool(ranges)
            full_count += fully_covered(ranges, end-start)
    return {"evidence_recall": any_count / len(aligned),
            "full_evidence_recall": full_count / len(aligned),
            "all_evidence_covered": float(full_count == len(aligned))}


def retrieval_metrics(all_chunks, ranked, aligned, k):
    if k < 1:
        raise ValueError("Positive K required")
    flags = [evidence_coverage([c], aligned)["evidence_recall"] > 0 for c in ranked[:k]]
    relevant = sum(evidence_coverage([c], aligned)["evidence_recall"] > 0 for c in all_chunks)
    ideal = sum(1/math.log2(i+2) for i in range(min(k, relevant)))
    dcg = sum(int(hit)/math.log2(i+2) for i, hit in enumerate(flags))
    first = next((i for i, hit in enumerate(flags, 1) if hit), None)
    return {"hit": float(any(flags)), "precision": sum(flags)/k,
            "mrr": 1/first if first else 0., "ndcg": dcg/ideal if ideal else 0.,
            **evidence_coverage(ranked[:k], aligned)}
