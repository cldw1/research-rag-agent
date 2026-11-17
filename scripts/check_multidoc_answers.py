"""Verify the frozen multi-document answer comparison without model calls."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.multidoc_answers import build_report
from src.evaluation.pdf_answers import read_json


def check_answers(run, report_path, root=ROOT):
    saved = read_json(report_path)
    replayed, _ = build_report(run, root, Path(run) / 'ai_reviews.json')
    if replayed != saved:
        raise ValueError('Saved comparison differs from verified answers and source reviews')
    return {'answer_count': saved['answer_count'], 'report_verified': True,
            'model_calls': 0, 'review_status': saved['review_status']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--report', type=Path, default=ROOT / 'results/multidoc_answer_comparison.json')
    args = parser.parse_args()
    print(check_answers(args.run, args.report))


if __name__ == '__main__':
    main()
