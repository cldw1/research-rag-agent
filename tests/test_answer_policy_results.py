"""Public paired results must reconcile with reviews, without human-score claims."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('version', [1, 2])
def test_all_paired_results_and_negative_findings_are_retained(version):
    report = json.loads((ROOT/f'results/answer_policy_v{version}_comparison.json').read_text(encoding='utf-8'))
    assert report['answer_count'] == 12 and len(report['question_ids']) == 6
    assert report['total_generation']['status_counts'] == {'succeeded': 12}
    assert report['independent_human_metrics'] is None
    assert report['review_provenance']['reviewer_type'] == 'ai_source_audit'
    assert report['review_provenance']['independent_human_review'] is False
    assert report['answer_policy']['version'] == report['policy_version'] == f'scientific-grounding-v{version}'
    by_id = {r['question_id']: r for r in report['review_records']}
    assert len(by_id) == 12 and set(by_id) == set(report['result_sha256'])
    for variant in ('legacy', 'grounded'):
        rows = [by_id[q+'--'+variant] for q in report['question_ids']]
        quality = report['variants'][variant]['ai_source_review']
        assert quality['complete'] and quality['human_metrics'] is None
        for metric, counts in quality['metrics'].items():
            for score in range(3):
                assert counts[f'score_{score}'] == sum(r[metric] == score for r in rows)
        assert all(r['paper_attribution_reason'].strip() for r in rows)
    for metric, changes in report['paired_changes'].items():
        deltas = [by_id[q+'--grounded'][metric]-by_id[q+'--legacy'][metric] for q in report['question_ids']]
        assert changes == {'improved': sum(d>0 for d in deltas), 'unchanged': sum(d==0 for d in deltas),
                           'regressed': sum(d<0 for d in deltas)}
    usage = report['total_generation']['known_returned_usage']
    assert usage['total_tokens'] == usage['prompt_tokens']+usage['completion_tokens']
    for key, total in usage.items():
        assert sum(v['known_returned_usage'][key] for v in report['variants'].values()) == total
    assert report['paired_changes']['faithfulness']['regressed'] > 0
    assert 'answer' not in report and 'messages' not in report


@pytest.mark.parametrize('version', [1, 2])
def test_saved_versions_replay_even_after_the_live_policy_changes(version):
    from scripts.run_answer_policy_check import build_report
    public = json.loads((ROOT/f'results/answer_policy_v{version}_comparison.json').read_text(encoding='utf-8'))
    run = ROOT/'data/pdf_benchmark/answer_policy_runs'/public['run_id']
    if not (run/'ai_reviews.json').exists():
        pytest.skip('Optional local frozen API responses')
    rebuilt, _ = build_report(run, run/'ai_reviews.json')
    assert rebuilt == public
