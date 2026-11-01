"""Reproduce section retrieval with the published questions and core retriever.

Requires the exact HMC PDF. No LLM calls; downloads require --allow-download.
Historical manual evidence ratings are never copied into the new experiment.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def evaluate(retriever, questions):
    from src.retrieval.retrieval_strategy import retrieve_with_strategy

    records = []
    for question in questions:
        record = {k: question[k] for k in ("id", "question", "gold_section", "gold_section_title")}
        for strategy in ("hybrid", "section_aware"):
            ranked = retrieve_with_strategy(retriever, question["question"], retriever.paper_id, strategy=strategy)
            gold_rank = None
            top5 = []
            for rank, chunk in enumerate(ranked, 1):
                match = re.match(r"^(\d+(?:\.\d+)+)(?![\d.])", chunk["section_name"].strip())
                if gold_rank is None and match and match.group(1) == question["gold_section"]:
                    gold_rank = rank
                top5.append({"rank": rank, "chunk_id": chunk["chunk_id"], "page": chunk["page_number"], "section": chunk["section_name"]})
            record[strategy] = {"gold_rank": gold_rank, "top5": top5}
        records.append(record)
    if not records:
        raise ValueError("Question list is empty")
    metrics = {}
    for strategy in ("hybrid", "section_aware"):
        ranks = [r[strategy]["gold_rank"] for r in records]
        metrics[strategy] = {
            "hit_at_5": sum(r is not None for r in ranks) / len(ranks),
            "mrr_at_5": sum(1 / r if r else 0 for r in ranks) / len(ranks),
        }
    return {"question_count": len(records), "metrics": metrics, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true", help="Allow BGE model download if not cached")
    args = parser.parse_args()

    from src.retrieval.verified_v3_pdf_retriever import VERIFIED_FILENAME, VERIFIED_PDF_SHA256, VerifiedV3PDFRetriever
    from src.retrieval.uploaded_pdf_retriever import MAX_PDF_BYTES

    if not args.pdf.is_file() or args.pdf.stat().st_size > MAX_PDF_BYTES:
        parser.error("需要存在且不超过 20 MB 的指定 PDF")
    pdf_bytes = args.pdf.read_bytes()
    if args.pdf.name != VERIFIED_FILENAME or hashlib.sha256(pdf_bytes).hexdigest() != VERIFIED_PDF_SHA256:
        parser.error("文件名或 SHA256 不匹配，不允许复用 HMC 章节映射")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5", device="cpu",
        cache_folder=str(ROOT / "models/embedding_cache"),
        local_files_only=not args.allow_download,
    )
    retriever = VerifiedV3PDFRetriever(pdf_bytes, args.pdf.name, model, ROOT / "results/pdf_v3_section_mapping_report.json")
    historical = json.loads((ROOT / "results/hmc_v3_public_summary.json").read_text(encoding="utf-8"))
    report = evaluate(retriever, historical["records"])
    report.update(pdf_sha256=VERIFIED_PDF_SHA256, page_count=retriever.page_count, chunk_count=len(retriever.chunks),
                  scope="Fresh retrieval only. No new manual evidence sufficiency or generated-answer evaluation.")
    output = ROOT / "results/hmc_v3_reproduced.local.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print("New retrieval report (historical summary unchanged):", output.name)


if __name__ == "__main__":
    main()
