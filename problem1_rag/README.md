# Problem 1 — Cost-Efficient RAG Application

A grounded QA service over PDF, HTML, and Markdown corpora backed by an embedded **SQLite + NumPy** vector store.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Ingest corpus (default chunk_size=500, overlap=100)
python3 -m rag_app.cli ingest --corpus rag_app/eval/corpus

# Query via CLI
python3 -m rag_app.cli query --question "How does SQLite prevent duplicate vectors?" --top-k 5

# Or run HTTP server
uvicorn rag_app.server:app --host 0.0.0.0 --port 8000
```

## Run Evaluation & Tests
```bash
python3 -m rag_app.eval.run_eval
python3 -m unittest discover -s tests
```

## Discussion
- **When to switch back to a managed DB**: Once the index exceeds single-node RAM (>10M vectors), requires multi-writer concurrency, or needs sub-10 ms p95 ANN search at >20 QPS via HNSW.
- **Weak link**: Retrieval quality on paraphrased queries when using lexical/hash embeddings (`local_tfidf`) vs dense neural embeddings (`text-embedding-3-small`); switching `EMBEDDING_PROVIDER=openai` resolves semantic paraphrase gaps while keeping the same SQLite store.
