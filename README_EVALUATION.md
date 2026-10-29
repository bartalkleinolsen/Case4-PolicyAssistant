# Case 4 — Systematic evaluation (English)

This adds a **reproducible 16-question test set**, with 8 questions whose answers are supported by company policies and 8 questions where the requested detail is absent. The app uses **the same three functions** as before. The only material `app.py` refactor is wrapping UI code in `main()` so that the runner can import the functions; the UI and model behavior are unchanged, except a metric is renamed from misleading "generation tokens" to "Total LLM tokens".

## Setup (VS Code / Windows)

1. Back up your existing `app.py` if you want to keep a copy. Extract this ZIP. Copy `app.py` and `run_evaluation.py` to the root `Case4-PolicyAssistant` folder, and the `tests/evaluation_cases.csv` file into the already existing `tests/` folder. Choose **Replace** only for `app.py`. Do NOT replace `.env`, `.gitignore`, `data/`, `rag_utils.py`, `build_index.py`, or `data/policy_vectors.json`.
2. Open **Terminal → New Terminal** in VS Code, keeping the Streamlit terminal running.
3. Run:

```powershell
.\.venv\Scripts\python.exe run_evaluation.py
```

4. The runner evaluates **3 questions per execution** and saves **each method result immediately** to `tests/evaluation_results.csv`. Run the exact same command again for the next three, or use `--limit 1` for smaller batches. `--all` runs every remaining case in one execution. Re-running skips successful results, so it will not repeat earlier API calls. On an error, it stops and retains progress.
5. Once complete, upload `tests/evaluation_results.csv` to your ChatGPT Case 4 thread for **human review** of correctness, relevance, and unsupported claims; do not automatically treat a policy-name match as proof of a grounded answer.

## Evaluation interpretation

- **Supported question:** a correct response must address the specific question and be supported by the identified policy. Merely quoting a related policy can be incomplete.
- **Unsupported question:** an answer should **explicitly abstain** from inventing the missing fact. A fact copied from a related policy that does not answer the question is incomplete, not necessarily hallucinated.
- **Hallucination:** a specific factual claim NOT substantiated by the company policies. This requires human assessment of the answer text against the entire CSV.
- **Policy-title match** is separately calculated in the CSV, but does **not** demonstrate accuracy or establish hallucination rates.
- **Tokens:** the LLM token metric excludes RAG's separate query embedding request and excludes the original one-time policy-indexing calls. RAG runtime includes its query embedding. Do not compare the 0 / ~3,700 / ~350 numbers as total combined service token usage.
- These 16 purposively selected cases provide a controlled teaching evaluation, **not** a statistically representative estimate of deployment error rates. Results can vary between runs.

## Security and hosting

Never publish or upload `.env` / API keys. Do not put evaluation runner behind a public button (it would allow arbitrary users to spend your API quota). We will display only selected evaluation summary results on the deployed website after reviewing the CSV.
