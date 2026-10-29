# Case 4 — Method 3 (Gemini embedding-based RAG)

This update **replaces `app.py`** and adds `rag_utils.py` and `build_index.py`. It does **not** replace `.env`, `.gitignore`, `data/company_policies.csv`, or the requirements.

## Start in the Case4-PolicyAssistant directory

1. Copy the three `.py` files into the project root (`Case4-PolicyAssistant`). Approve replacing `app.py` only.
2. Keep `.env` and `.gitignore` as they are.
3. Open **Terminal → New Terminal** in VS Code and run:

```powershell
.\.venv\Scripts\python.exe build_index.py
```

4. Wait for **SUCCESS: 98 policies indexed** (displayed at the end of the command). You should see a file `data/policy_vectors.json` created. If a quota/rate limit error occurs, re-run the **same command later**. Completed batches are saved and resume automatically.
5. Open the existing Streamlit browser tab, then press Ctrl+R (refresh). You should see 3 methods. If the app is no longer running, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

6. Test: `How many days can I work abroad?` → Work From Abroad Policy → up to 30 days with approval.

## Why this is a real vector index

`build_index.py` converts each CSV record into a **768-dimensional Gemini Embedding 2 vector** and saves it in JSON. On each question, Method 3 calls the same embedding model to encode the **question**, computes cosine similarity to the stored vectors, retrieves only the **4 nearest policies**, and passes those policies to the answer model. Method 2 passes all policies as plain text without retrieval; Method 1 does deterministic keyword overlap.

The time metric for Method 3 includes question embedding + retrieval + answer generation; the token metric only measures **answer-generation tokens**. The one-time indexing calls and query embedding are not represented by that metric. For the final report, we should disclose these separate retrieval costs and test unsupported questions as well as supported questions.

Google reference: https://ai.google.dev/gemini-api/docs/embeddings

## Next steps

Method 1, 2, 3 need systematic evaluation of policy identification, support/hallucination and runtime; then deployment, GitHub repository and Slack bot. Never commit `.env` or publish your key.
