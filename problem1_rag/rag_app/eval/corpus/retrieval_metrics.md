# Retrieval Evaluation Metrics: Recall@k, MRR, and nDCG@k

Evaluating a RAG system requires separating retrieval quality from generation quality. If the retriever fails to surface the right passage, even the strongest LLM cannot produce a grounded answer.

## Core Information Retrieval Metrics
- **Recall@k (Hit Rate)**: Measures the fraction of queries where at least one relevant chunk appears in the top-k results.
- **Mean Reciprocal Rank (MRR)**: Averages `1 / rank` of the first relevant chunk across all queries, rewarding systems that place the best evidence at rank 1.
- **nDCG@k**: Normalized Discounted Cumulative Gain accounts for both graded relevance and logarithmic position decay across all top-k slots.
- **Context Precision**: Measures the proportion of retrieved chunks in the top-k set that are actually relevant to the query, penalizing noisy context windows.
