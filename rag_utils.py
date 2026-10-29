"""Shared helpers for the real embedding-based RAG implementation (Case 4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "company_policies.csv"
INDEX_FILE = BASE_DIR / "data" / "policy_vectors.json"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


def dataset_hash() -> str:
    return hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()


def document_text(record: dict) -> str:
    """Embedding 2's recommended asymmetric document format."""
    return (
        f"title: {record['title']} | text: {record['policy_text']} "
        f"Department: {record['department']}. Category: {record['category']}."
    )


def query_text(question: str) -> str:
    """Embedding 2's recommended asymmetric question-answering query format."""
    return f"task: question answering | query: {question}"


def read_index() -> dict:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            "Vector index missing. Run: .\\.venv\\Scripts\\python.exe build_index.py"
        )
    with INDEX_FILE.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if index.get("dataset_sha256") != dataset_hash():
        raise ValueError("The CSV changed after indexing. Run build_index.py again.")
    if index.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("The embedding model changed. Rebuild the index.")
    if index.get("dimensions") != EMBEDDING_DIMENSIONS:
        raise ValueError("The vector dimensions changed. Rebuild the index.")
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) != index.get("policy_count"):
        raise ValueError("The vector index is incomplete. Rerun build_index.py.")
    if not entries or any(len(e.get("vector", [])) != EMBEDDING_DIMENSIONS for e in entries):
        raise ValueError("The vector index contains missing or invalid vectors.")
    return index


def top_matches(query_vector: list[float], entries: list[dict], k: int = 4) -> list[dict]:
    """Cosine-similarity retrieval. Only numeric vectors determine ranking."""
    matrix = np.asarray([entry["vector"] for entry in entries], dtype=np.float32)
    q = np.asarray(query_vector, dtype=np.float32)
    if matrix.shape[1] != q.size:
        raise ValueError("Embedding dimension mismatch.")
    lengths = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    scores = (matrix @ q) / np.maximum(lengths, 1e-12)
    best = np.argsort(-scores)[:k]
    return [{**{field: entry for field, entry in entries[int(i)].items() if field != "vector"},
             "similarity": float(scores[int(i)])} for i in best]
