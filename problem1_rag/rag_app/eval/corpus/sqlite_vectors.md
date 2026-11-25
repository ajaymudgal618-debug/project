# SQLite Vector Storage Architecture

SQLite is an embedded, serverless SQL database engine stored in a single cross-platform file. For lightly-queried vector indexes up to a few million embeddings, storing float32 vectors as BLOBs in SQLite alongside relational metadata eliminates always-on pod costs.

## Idempotent Ingestion and Deduplication
To prevent duplicate vectors during re-ingestion, each document is hashed with SHA-256 over its relative path, chunking parameters, and normalized text. Each chunk receives a deterministic content-addressed primary key combining the document ID, chunk index, and chunk hash. Using `INSERT OR IGNORE` and `ON CONFLICT` updates ensures re-running ingestion on an unchanged corpus inserts zero new vectors.

## Metadata Filtering
SQLite B-tree indexes on `source_type` and `topic` allow pre-filtering candidate chunks in SQL before loading float32 buffers into NumPy for exact cosine similarity scoring. At 100,000 vectors of 384 dimensions, a full float32 matrix occupies roughly 153.6 MB, enabling sub-20 ms brute-force dot-product search on commodity CPU instances without approximate index overhead.
