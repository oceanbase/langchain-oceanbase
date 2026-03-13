#!/usr/bin/env python3
"""
quickstart.py

Demonstrates:
- connecting to OceanBase via MySQL-compatible port
- creating a documents table
- generating embeddings via OpenAI
- storing embeddings in the DB (as JSON)
- performing a simple in-Python nearest-neighbor search

Requirements:
pip install openai mysql-connector-python numpy
Set env vars: OB_HOST, OB_PORT, OB_USER, OB_PASSWORD, OB_DB, OPENAI_API_KEY
"""
import os
import json
import time
import mysql.connector
import openai
import numpy as np

# Configuration (from environment)
OB_HOST = os.environ.get("OB_HOST", "127.0.0.1")
OB_PORT = int(os.environ.get("OB_PORT", 3306))
OB_USER = os.environ.get("OB_USER", "root")
OB_PASSWORD = os.environ.get("OB_PASSWORD", "")
OB_DB = os.environ.get("OB_DB", "langchain_ob_demo")

openai.api_key = os.environ.get("OPENAI_API_KEY", "")

def get_db_conn():
    return mysql.connector.connect(
        host=OB_HOST, port=OB_PORT, user=OB_USER, password=OB_PASSWORD, database=OB_DB
    )

def init_db():
    conn = mysql.connector.connect(host=OB_HOST, port=OB_PORT, user=OB_USER, password=OB_PASSWORD)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{OB_DB}`;")
    conn.commit()
    cursor.close()
    conn.close()

    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """
    )
    conn.commit()
    cursor.close()
    conn.close()

def embed_texts(texts):
    # Uses OpenAI embeddings (text-embedding-3-small as an example).
    # You can replace with another embeddings provider.
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate embeddings for this demo")
    resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
    embeddings = [r["embedding"] for r in resp["data"]]
    return embeddings

def insert_document(content, embedding):
    conn = get_db_conn()
    cursor = conn.cursor()
    emb_json = json.dumps(embedding)
    cursor.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (content, emb_json))
    conn.commit()
    cursor.close()
    conn.close()

def fetch_all_docs():
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, content, embedding FROM documents")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    results = []
    for r in rows:
        emb = json.loads(r["embedding"]) if r["embedding"] else None
        results.append({"id": r["id"], "content": r["content"], "embedding": emb})
    return results

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def simple_search(query, top_k=3):
    q_emb = embed_texts([query])[0]
    docs = fetch_all_docs()
    scored = []
    for d in docs:
        if d["embedding"]:
            score = cosine_similarity(q_emb, d["embedding"])
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def main():
    print("Initializing DB...")
    init_db()
    time.sleep(1)

    # Insert some demo docs if table empty
    docs = [
        "Python is a popular programming language for data science and web development.",
        "OceanBase is a distributed relational database compatible with MySQL.",
        "LangChain helps glue together language models and your data in vectors.",
    ]
    existing = fetch_all_docs()
    if not existing:
        print("Inserting demo documents and embeddings...")
        embeddings = embed_texts(docs)
        for txt, emb in zip(docs, embeddings):
            insert_document(txt, emb)
        print("Inserted demo documents.")

    query = "How can I use vectors with databases?"
    print(f"Searching for: {query}")
    results = simple_search(query)
    for score, doc in results:
        print(f"\nScore: {score:.4f}\nDoc id: {doc['id']}\n{doc['content']}")

if __name__ == "__main__":
    main()