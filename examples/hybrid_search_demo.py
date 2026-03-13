#!/usr/bin/env python3
"""
hybrid_search_demo.py

Demonstrates a hybrid search approach:
 - Run a MySQL full-text search to get candidate documents
 - Compute embedding similarity for candidates and re-rank
This is useful when combining sparse (keyword) and dense (embedding) signals.

Requirements:
pip install openai mysql-connector-python numpy
Set env vars: OB_HOST, OB_PORT, OB_USER, OB_PASSWORD, OB_DB, OPENAI_API_KEY
"""
import os
import json
import mysql.connector
import openai
import numpy as np
from examples.quickstart import get_db_conn, embed_texts, cosine_similarity  # relative import

openai.api_key = os.environ.get("OPENAI_API_KEY", "")

def ft_search_mysql(query, limit=10):
    """
    Performs a simple full-text search using MATCH ... AGAINST.
    Make sure the 'documents' table has a fulltext index on content if supported:
    ALTER TABLE documents ADD FULLTEXT(content);
    Note: Some MySQL variants or OceanBase setups may require different syntax or engine support.
    """
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    # Use natural language fulltext search
    try:
        cursor.execute(
            "SELECT id, content, embedding, MATCH(content) AGAINST (%s IN NATURAL LANGUAGE MODE) as ft_score "
            "FROM documents WHERE MATCH(content) AGAINST (%s IN NATURAL LANGUAGE MODE) LIMIT %s",
            (query, query, limit),
        )
        rows = cursor.fetchall()
    except Exception:
        # Fallback to LIKE if fulltext not available
        cursor.execute(
            "SELECT id, content, embedding, 0 as ft_score FROM documents WHERE content LIKE %s LIMIT %s",
            (f"%{query}%", limit),
        )
        rows = cursor.fetchall()
    cursor.close()
    conn.close()
    results = []
    for r in rows:
        emb = json.loads(r["embedding"]) if r.get("embedding") else None
        results.append({"id": r["id"], "content": r["content"], "embedding": emb, "ft_score": r.get("ft_score", 0)})
    return results

def hybrid_search(query, top_k=5, alpha=0.5):
    """
    alpha: weight for dense score. hybrid_score = alpha * dense + (1-alpha) * normalized_ft
    """
    candidates = ft_search_mysql(query, limit=50)
    if not candidates:
        return []

    q_emb = embed_texts([query])[0]
    dense_scores = []
    for c in candidates:
        if c["embedding"]:
            dense_scores.append(cosine_similarity(q_emb, c["embedding"]))
        else:
            dense_scores.append(0.0)

    # normalize ft scores and dense scores
    ft_scores = np.array([float(c.get("ft_score") or 0.0) for c in candidates], dtype=float)
    dense_scores = np.array(dense_scores, dtype=float)

    def normalize(arr):
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    nf = normalize(ft_scores)
    nd = normalize(dense_scores)
    hybrid = alpha * nd + (1 - alpha) * nf

    combined = []
    for i, c in enumerate(candidates):
        combined.append((hybrid[i], c, float(nd[i]), float(nf[i])))
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined[:top_k]

def main():
    query = "distributed relational database"
    print(f"Hybrid search for: {query}")
    results = hybrid_search(query, top_k=5, alpha=0.6)
    for score, doc, dense_score, ft_score in results:
        print(f"\nHybrid score: {score:.4f} (dense: {dense_score:.4f}, ft: {ft_score:.4f})")
        print(f"Doc id: {doc['id']}\n{doc['content'][:250]}")

if __name__ == "__main__":
    main()