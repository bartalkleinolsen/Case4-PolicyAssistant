# Case 4 – Evaluation dashboard update

This is an update to the **existing** Case4-PolicyAssistant project. It does not replace your policy database, saved vector index, or API secrets.

## Files to copy

1. `app.py` → replace the existing `Case4-PolicyAssistant/app.py`.
2. `evaluation_metrics.py` → copy to `Case4-PolicyAssistant/evaluation_metrics.py` (new file).
3. `tests/evaluation_review.csv` → copy to the existing project's `tests` folder (new file).

**Keep your existing:** `.env`, `.gitignore`, `data/company_policies.csv`, `data/policy_vectors.json`, `rag_utils.py`, `build_index.py`, `run_evaluation.py`, and `tests/evaluation_results.csv`.

IMPORTANT: The dashboard reads the original evaluation results from the user's existing `tests/evaluation_results.csv`, so **that file must remain in the tests folder**. The archive deliberately does not replace that file.

## Run

If Streamlit is already running, refresh `http://localhost:8501` using `Ctrl+R`; otherwise, in the Case4-PolicyAssistant terminal run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Scroll under the three methods to the section **Evaluation: 16 Fixed Questions**. The section displays one summary table, reviewer-methodology notes, and a dropdown to inspect each of the 48 stored responses without making extra API calls.

## How the metrics are calculated

- **Correct policy (8 answerable questions):** exact match between recorded `returned_policy` and `expected_policy` in `evaluation_results.csv`.
- **Missing detail recognised (8 unanswerable questions):** separately reviewed yes/no flag in `evaluation_review.csv` for an explicit admission that the requested fact is unavailable. Returning a related policy without explicitly saying the requested information is missing does not count.
- **Clear invented policy facts (all 16 questions):** human review based on recorded answers. No clear invented policy facts were observed in these 48 responses; other failure types (such as an irrelevant policy or an incomplete answer) are not counted as fabricated facts. This small test is *not* a reliable estimate of population hallucination rates.
- **Response time and LLM tokens:** per-method means of recorded values for all 16 questions, no extra API requests.
- **RAG tokens:** exclude query embeddings and initial creation of the vector index; RAG response time includes query embedding.

### Notable observed examples
- S07: Rules-Based returned `Business Continuity Policy` rather than `Travel Reimbursement Policy`.
- U02: Rules-Based returned `Parking Policy` but did not say the free-parking question was unanswered; the two LLM approaches did.
- U03 and U06: The no-vector LLM returned related policy statements without explicitly acknowledging the missing numeric detail.
- S01 and S05: RAG made the correct policy identification but hedged when applying an explicit numerical limit.

This update does not make the app public. A separate GitHub and Streamlit Community Cloud deployment is still required, and secrets must not be committed.
