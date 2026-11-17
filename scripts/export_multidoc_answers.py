"""Replay a complete multi-document source audit without any model requests."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.multidoc_answers import build_report, write_human_review_sheet
from src.evaluation.pdf_answers import read_json, write_new


def export_answers(run, reviews_file, output, root=ROOT):
    run, output = Path(run), Path(output)
    report, pack = build_report(run, root, reviews_file)
    if any(row['status'] != 'succeeded' for row in pack):
        raise ValueError('Public comparison requires all 24 answers to have succeeded')
    # The report contains numeric diagnostics and review reasons, not full passages.
    write_new(output, report)
    sheet = run / 'human_multidoc_answer_review.csv'
    if not sheet.exists():
        write_human_review_sheet(sheet, pack, read_json(reviews_file))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = export_answers(args.run, args.reviews, args.output)
    print({'answer_count': report['answer_count'], 'review_status': report['review_status'],
           'paired_correctness': {k: v for k, v in report['paired_correctness'].items() if k != 'records'}})


if __name__ == '__main__':
    main()
