#!/usr/bin/env python3
"""
rag_demo.py

A simple RAG (Retrieve-and-Generate) demo:
- Retrieve top-k documents by embedding similarity (using the same DB approach as quickstart)
- Send retrieved documents + question to OpenAI ChatCompletion to generate an answer

Requirements:
pip install openai mysql-connector-python numpy
Set env vars: OB_HOST, OB_PORT, OB_USER, OB_PASSWORD, OB_DB, OPENAI_API_KEY
"""
import os
import json
import mysql.connector
import openai
import numpy as np
from examples.quickstart import get_db_conn, fetch_all_docs, embed_texts, cosine_similarity  # relative import

openai.api_key = os.environ.get("OPENAI_API_KEY", "")

def retrieve(query, top_k=3):
    q_emb = embed_texts([query])[0]
    docs = fetch_all_docs()
    scored = []
    for d in docs:
        if d["embedding"]:
            score = cosine_similarity(q_emb, d["embedding"])
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k]]

def generate_answer(query, context_docs):
    # Compose a prompt for the LLM using retrieved docs
    system = {"role": "system", "content": "You are a helpful assistant."}
    context_text = "\n\n".join([f"Document {d['id']}: {d['content']}" for d in context_docs])
    prompt = (
        f"Use the following documents as context to answer the user's question.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer concisely with references to the documents when useful."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system, {"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()

def main():
    question = "What is OceanBase and how does it relate to vector storage?"
    print("Retrieving relevant docs...")
    docs = retrieve(question, top_k=3)
    print("Retrieved docs:")
    for d in docs:
        print(f"- (id={d['id']}) {d['content'][:120]}")

    print("\nGenerating answer with LLM...")
    answer = generate_answer(question, docs)
    print("\nAnswer:\n")
    print(answer)

if __name__ == "__main__":
    main()