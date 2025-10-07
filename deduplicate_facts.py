#!/usr/bin/env python3
"""CLI entry point for fact deduplication."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deduplicate.core import DeduplicationConfig, DeduplicationOrchestrator
from ie.client import LlamaServerConfig


def parse_args() -> argparse.Namespace:
    default_llm = LlamaServerConfig()
    parser = argparse.ArgumentParser(description="Deduplicate IE facts in SQLite.")
    parser.add_argument("--sqlite", type=Path, default=Path("./discord.db"), help="Path to SQLite database.")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j bolt URI.")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password.")
    parser.add_argument("--minhash-threshold", type=float, default=0.80, help="Jaccard threshold for MinHash LSH.")
    parser.add_argument("--minhash-num-perm", type=int, default=128, help="Number of MinHash permutations.")
    parser.add_argument("--minhash-ngram-size", type=int, default=3, help="Character n-gram size for MinHash tokens.")
    parser.add_argument("--embedding-model", default="google/embeddinggemma-300m", help="SentenceTransformer embedding model name.")
    parser.add_argument("--embedding-threshold", type=float, default=0.95, help="Cosine similarity threshold for embeddings.")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding encoder batch size.")
    parser.add_argument("--embedding-device", help="Force embedding model device (e.g. cuda or cpu).", default="cpu")
    parser.add_argument("--embedding-max-neighbors", type=int, default=25, help="Maximum embedding neighbors per fact before dedup union.")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum fact confidence to consider for deduplication.")
    parser.add_argument("--max-partitions", type=int, help="Process at most N partitions (useful for testing).")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent running/paused deduplication run.")
    parser.add_argument("--dry-run", action="store_true", help="Run without persisting changes or updating Neo4j.")
    parser.add_argument("--graph-delete-batch-size", type=int, default=100, help="Batch size when deleting old Neo4j relationships.")
    parser.add_argument("--max-group-size", type=int, default=10, help="Maximum fact count sent to the LLM per candidate group.")
    parser.add_argument("--llm-base-url", default=default_llm.base_url, help="LLM endpoint base URL.")
    parser.add_argument("--llm-model", default=default_llm.model, help="LLM model name.")
    parser.add_argument("--llm-timeout", type=float, default=default_llm.timeout, help="LLM request timeout in seconds.")
    parser.add_argument("--llm-temperature", type=float, default=default_llm.temperature, help="LLM sampling temperature.")
    parser.add_argument("--llm-top-p", type=float, default=default_llm.top_p, help="LLM nucleus sampling top_p.")
    parser.add_argument("--llm-max-tokens", type=int, default=default_llm.max_tokens, help="LLM max token count.")
    parser.add_argument("--llm-api-key", help="Optional API key for the LLM service.")
    parser.add_argument("--llm-max-retries", type=int, default=3, help="Max LLM retries per candidate group.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    llm_config = LlamaServerConfig(
        base_url=args.llm_base_url,
        model=args.llm_model,
        timeout=args.llm_timeout,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_tokens=args.llm_max_tokens,
        api_key=args.llm_api_key,
    )

    config = DeduplicationConfig(
        sqlite_path=args.sqlite,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        minhash_threshold=args.minhash_threshold,
        minhash_num_perm=args.minhash_num_perm,
        minhash_ngram_size=args.minhash_ngram_size,
        embedding_model=args.embedding_model,
        embedding_threshold=args.embedding_threshold,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        embedding_max_neighbors=args.embedding_max_neighbors,
        llm_config=llm_config,
        llm_max_retries=args.llm_max_retries,
        min_confidence=args.min_confidence,
        resume=args.resume,
        dry_run=args.dry_run,
        max_partitions=args.max_partitions,
        graph_delete_batch_size=args.graph_delete_batch_size,
        max_group_size=args.max_group_size,
    )

    orchestrator = DeduplicationOrchestrator(config)
    stats = orchestrator.run()
    print(
        f"Deduplication run {stats.run_id} finished in {stats.elapsed_time:.1f}s; "
        f"processed {stats.processed_partitions}/{stats.total_partitions} partitions, "
        f"merged {stats.facts_merged} facts across {stats.candidate_groups_processed} groups."
    )


if __name__ == "__main__":
    main()
