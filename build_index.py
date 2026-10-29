"""Embed all 98 company policies ONCE, save a resumable JSON vector index.

Run from the Case4-PolicyAssistant directory:
    .\\.venv\\Scripts\\python.exe build_index.py

If Gemini rate-limits you, completed batches are already saved. Retry later.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag_utils import (
    BASE_DIR,
    DATA_FILE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    INDEX_FILE,
    dataset_hash,
    document_text,
)

load_dotenv(BASE_DIR / ".env")
BATCH_SIZE = 10  # Low request size; saves each successful batch.


def save_progress(payload: dict) -> None:
    temporary_file = Path(str(INDEX_FILE) + ".tmp")
    with temporary_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary_file.replace(INDEX_FILE)


def main() -> int:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "indsæt_din_nøgle_her":
        print("ERROR: Please enter your Gemini API key in .env first.")
        return 1

    policies = pd.read_csv(DATA_FILE, keep_default_na=False)
    fields = ["title", "department", "policy_text", "category"]
    if not all(field in policies.columns for field in fields):
        print("ERROR: Missing policy columns in CSV.")
        return 1

    records = policies[fields].to_dict(orient="records")
    fingerprint = dataset_hash()
    header = {
        "dataset_sha256": fingerprint,
        "embedding_model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "policy_count": len(records),
    }
    payload = {**header, "entries": []}

    if INDEX_FILE.exists():
        try:
            existing = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            if all(existing.get(k) == v for k, v in header.items()):
                saved = existing.get("entries", [])
                if all(
                    saved[i]["title"] == records[i]["title"]
                    and len(saved[i]["vector"]) == EMBEDDING_DIMENSIONS
                    for i in range(min(len(saved), len(records)))
                ) and len(saved) <= len(records):
                    payload["entries"] = saved
                    print(f"Found existing index: {len(saved)}/{len(records)} policies.")
        except (ValueError, KeyError, TypeError, OSError):
            print("Existing index invalid; will rebuild it.")

    total = len(records)
    if len(payload["entries"]) == total:
        print(f"DONE: {total} policies already indexed. No API calls needed.")
        return 0

    print(f"Embedding {total} policies with {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dimensions).")
    print("Each successful batch is saved in data/policy_vectors.json.")
    client = genai.Client(api_key=key)

    try:
        while len(payload["entries"]) < total:
            start = len(payload["entries"])
            chunk = records[start : start + BATCH_SIZE]
            contents = [
                types.Content(parts=[types.Part.from_text(text=document_text(item))])
                for item in chunk
            ]
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(chunk):
                raise RuntimeError(
                    f"Expected {len(chunk)} separate vectors, got {len(embeddings)}. "
                    "Stopping to prevent an incorrect index."
                )
            for record, embedding in zip(chunk, embeddings):
                vector = embedding.values or []
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise RuntimeError("Unexpected embedding dimensions; check model settings.")
                payload["entries"].append({**record, "vector": list(map(float, vector))})
            save_progress(payload)
            print(f"Saved {len(payload['entries'])}/{total} policies.", flush=True)
    except Exception as exc:
        error = str(exc).replace(key, "[HIDDEN API KEY]")[:650]
        print(f"ERROR: {type(exc).__name__}: {error}")
        print(f"Completed: {len(payload['entries'])}/{total}. Rerun this command to resume.")
        return 1
    finally:
        client.close()

    print(f"SUCCESS: {total} policies indexed -> {INDEX_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
