"""Reproducible aggregated metrics for the 16-question Case 4 evaluation.

Only uses Python standard library. No API calls and no access to .env.
Reviewer-coded flags live separately in tests/evaluation_review.csv.
"""
import csv
from pathlib import Path
from statistics import mean

METHODS = ('Rules-Based', 'LLM (No Vectors)', 'LLM + RAG')


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open('r', newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def get_evaluation_summary(results_path: Path, review_path: Path) -> dict:
    results = _read_csv(results_path)
    reviews = _read_csv(review_path)
    if not results or not reviews:
        raise ValueError('No evaluation data found.')

    by_key = {}
    for row in results:
        key = (row['case_id'], row['method'])
        if key in by_key:
            raise ValueError(f'Duplicate evaluation result: {key}')
        if row.get('error', '').strip():
            raise ValueError(f'Evaluation contains an error for {key}')
        by_key[key] = row

    reviewer_by_key = {}
    for row in reviews:
        key = (row['case_id'], row['method'])
        if key in reviewer_by_key:
            raise ValueError(f'Duplicate manual review: {key}')
        reviewer_by_key[key] = row
    if set(by_key) != set(reviewer_by_key):
        raise ValueError('Manual review keys do not match evaluation results.')

    all_cases = {r['case_id']: r['answerable'].lower() for r in results}
    supported = {cid for cid, a in all_cases.items() if a == 'yes'}
    unsupported = {cid for cid, a in all_cases.items() if a == 'no'}
    if len(all_cases) != 16 or len(supported) != 8 or len(unsupported) != 8:
        raise ValueError('Expected exactly 16 cases: 8 answerable and 8 unsupported.')
    if len(by_key) != 16*len(METHODS):
        raise ValueError('Expected 48 method results (16 questions x 3 methods).')

    metrics = {}
    for method in METHODS:
        mrows = [row for row in results if row['method'] == method]
        if len(mrows) != 16:
            raise ValueError(f'{method} does not have exactly 16 records.')
        sc = [r for r in mrows if r['case_id'] in supported]
        uc = [r for r in mrows if r['case_id'] in unsupported]
        correct_policy = sum(
            r['returned_policy'].strip() == r['expected_policy'].strip()
            for r in sc
        )
        recognized_missing = sum(
            reviewer_by_key[(r['case_id'], method)]['recognises_missing_information'].lower() == 'yes'
            for r in uc
        )
        invented_claims = sum(
            reviewer_by_key[(r['case_id'], method)]['clear_invented_policy_fact'].lower() == 'yes'
            for r in mrows
        )
        for r in mrows:
            review = reviewer_by_key[(r['case_id'], method)]
            if review['clear_invented_policy_fact'].lower() not in ('yes','no'):
                raise ValueError('Reviewer flags for invented facts must be yes/no.')
            if r['case_id'] in unsupported and review['recognises_missing_information'].lower() not in ('yes','no'):
                raise ValueError('Unsupported-case reviewer flags must be yes/no.')

        metrics[method] = {
            'matched_policy': correct_policy,
            'missing_information_recognised': recognized_missing,
            'missing_information_not_recognised': len(unsupported) - recognized_missing,
            'invented_policy_facts': invented_claims,
            'mean_time_ms': mean(float(r['time_ms']) for r in mrows),
            'mean_llm_tokens': mean(int(r['llm_total_tokens']) for r in mrows),
            'mean_embedding_calls': mean(int(r['query_embedding_calls']) for r in mrows),
        }
    return {'metrics': metrics, 'questions': len(all_cases),
            'supported_questions': len(supported),
            'unsupported_questions': len(unsupported)}
