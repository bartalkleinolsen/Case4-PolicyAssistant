r"""Run pre-defined evaluation questions against the SAME methods as app.py.

From project root (Windows PowerShell):
 .\.venv\Scripts\python.exe run_evaluation.py

Default is only 3 incomplete questions per run. Completed individual method
results are saved after each call and are never re-run unless --restart is used.
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from app import DATA_FILE, INDEX_FILE, llm_without_vectors, rag_answer, rule_based_search

BASE_DIR = Path(__file__).resolve().parent
CASES = BASE_DIR / 'tests' / 'evaluation_cases.csv'
OUTPUT = BASE_DIR / 'tests' / 'evaluation_results.csv'
METHODS = ('Rules-Based', 'LLM (No Vectors)', 'LLM + RAG')
FIELDS = [
    'case_id', 'question', 'answerable', 'expected_policy', 'expected_fact',
    'focus', 'method', 'answer', 'returned_policy', 'time_ms',
    'llm_input_tokens', 'llm_output_tokens', 'llm_total_tokens',
    'query_embedding_calls', 'retrieved_policies', 'title_exact_match',
    'error', 'timestamp_utc',
]


def load_csv(file: Path) -> list[dict]:
    if not file.exists():
        return []
    with file.open('r', newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def save_csv(rows: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, OUTPUT)


def evaluate(case: dict, method: str, policies: pd.DataFrame, key: str) -> dict:
    if method == 'Rules-Based':
        result = rule_based_search(case['question'], policies)
    elif method == 'LLM (No Vectors)':
        result = llm_without_vectors(case['question'], policies, key)
    else:
        result = rag_answer(case['question'], key)
    retrieved = result.get('retrieved', [])
    returned_title = result.get('policy', '')
    status_ok = 'error' not in result
    return {
        **case,
        'method': method,
        'answer': result.get('answer', ''),
        'returned_policy': returned_title,
        'time_ms': f"{result['time_ms']:.2f}" if status_ok else '',
        'llm_input_tokens': result.get('input_tokens', 0) or 0,
        'llm_output_tokens': result.get('output_tokens', 0) or 0,
        'llm_total_tokens': result.get('total_tokens', 0) or 0,
        'query_embedding_calls': 1 if method == 'LLM + RAG' else 0,
        'retrieved_policies': json.dumps([x['title'] for x in retrieved], ensure_ascii=False),
        'title_exact_match': 'yes' if status_ok and returned_title == case['expected_policy'] else 'no',
        'error': result.get('error', ''),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Resumable Case 4 evaluation, 3 questions per run by default.')
    parser.add_argument('--limit', type=int, default=3, help='Number of incomplete questions in this batch (default: 3).')
    parser.add_argument('--all', action='store_true', help='Run all remaining questions; uses more API calls.')
    parser.add_argument('--restart', action='store_true', help='Discard previous results and start again (costs more).')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be >= 1')
    if not DATA_FILE.exists() or not CASES.exists():
        parser.error('Missing data/company_policies.csv or tests/evaluation_cases.csv.')
    if not INDEX_FILE.exists():
        parser.error('Missing data/policy_vectors.json: run build_index.py first.')
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if not key or key == 'indsæt_din_nøgle_her':
        parser.error('Missing GEMINI_API_KEY. Keep it in .env, never share it.')

    policies = pd.read_csv(DATA_FILE, keep_default_na=False)
    cases = load_csv(CASES)
    if not cases:
        parser.error('No evaluation questions found.')
    rows = [] if args.restart else load_csv(OUTPUT)
    successful = {(r['case_id'], r['method']) for r in rows if not r.get('error')}
    incomplete = [c for c in cases if any((c['case_id'], m) not in successful for m in METHODS)]
    selected = incomplete if args.all else incomplete[:args.limit]
    if not selected:
        print('ALL DONE: No incomplete evaluation cases.')
        print(f'Results: {OUTPUT}')
        return
    print(f'Starting {len(selected)} of {len(incomplete)} incomplete questions ({len(cases)} total).')
    print('Each successful method result is saved to CSV immediately.')
    print('LLM calls can incur usage or cost; query embedding is additional to LLM tokens.')
    for case in selected:
        for method in METHODS:
            identity = (case['case_id'], method)
            if identity in successful:
                continue
            print(f"{case['case_id']} | {method} ...", flush=True)
            try:
                result = evaluate(case, method, policies, key)
            except Exception as exc:
                result = {
                    **case, 'method': method, 'answer': '', 'returned_policy': '',
                    'error': f'{type(exc).__name__}: {str(exc).replace(key, "[HIDDEN API KEY]")[:450]}',
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                }
            rows = [r for r in rows if (r['case_id'], r['method']) != identity]
            rows.append(result)
            save_csv(rows)
            if result.get('error'):
                print('ERROR: '+result['error'])
                print('Saved previous progress. Resolve the error, then rerun the same command.')
                return
            successful.add(identity)
            print(f"  Policy: {result['returned_policy']} | LLM tokens: {result['llm_total_tokens']} | {result['time_ms']} ms")
    complete = sum(all((c['case_id'], m) in successful for m in METHODS) for c in cases)
    print(f'PROGRESS: {complete}/{len(cases)} questions fully evaluated.')
    print(f'CSV saved: {OUTPUT}')
    if complete == len(cases):
        print('DONE. Upload evaluation_results.csv for manual answer-grounding review.')
    else:
        print('Run the same command again for the next 3 questions.')


if __name__ == '__main__':
    main()
