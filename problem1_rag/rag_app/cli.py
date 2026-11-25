"""CLI entrypoint for ingesting a corpus, querying, and inspecting store stats."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import get_config
from .ingest import ingest_directory
from .qa import answer_question
from .store import SQLiteVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-Efficient RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest PDF/HTML/MD folder")
    p_ing.add_argument("--corpus", required=True, help="Path to corpus folder")
    p_ing.add_argument("--chunk-size", type=int, default=None)
    p_ing.add_argument("--overlap", type=int, default=None)

    p_q = sub.add_parser("query", help="Run a RAG query")
    p_q.add_argument("--question", required=True)
    p_q.add_argument("--top-k", type=int, default=None)
    p_q.add_argument("--source-type", default=None, help="Filter: pdf, html, or md")
    p_q.add_argument("--topic", default=None, help="Filter by topic metadata")

    sub.add_parser("stats", help="Show vector store stats")

    args = parser.parse_args()
    cfg = get_config()

    if args.cmd == "ingest":
        out = ingest_directory(
            args.corpus, cfg=cfg, chunk_size=args.chunk_size, overlap=args.overlap
        )
        print(json.dumps(out, indent=2))
    elif args.cmd == "query":
        res = answer_question(
            args.question,
            cfg=cfg,
            top_k=args.top_k,
            source_type=args.source_type,
            topic=args.topic,
        )
        print(json.dumps(asdict(res), indent=2))
    elif args.cmd == "stats":
        print(json.dumps(SQLiteVectorStore(cfg.db_path).stats(), indent=2))


if __name__ == "__main__":
    main()
