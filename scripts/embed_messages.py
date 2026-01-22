#!/usr/bin/env python3
"""Generate semantic embeddings for Discord Message nodes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from neo4j import GraphDatabase

# Ensure repository root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory_agent.config import Settings  # noqa: E402
from memory_agent.embeddings import EmbeddingProvider  # noqa: E402
from memory_agent.message_embedding_pipeline import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    run_message_embedding_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("embed_messages")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate Neo4j message embeddings for semantic search."
    )
    parser.add_argument(
        "--settings-from-env", action="store_true", default=True, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Delete embeddings for messages that no longer exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute embeddings without writing them back to Neo4j.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embedding batch size (default from settings).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for CPU (default: 1, recommended: 2-4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    job_settings = settings.message_embeddings
    logger.info("Starting message embedding pipeline using model %s", job_settings.model)

    provider = EmbeddingProvider(
        model_name=job_settings.model,
        device=job_settings.device,
        cache_dir=job_settings.cache_dir,
    )

    driver = GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password),
        max_connection_lifetime=settings.neo4j.max_connection_lifetime,
        encrypted=settings.neo4j.encrypted,
    )
    batch_size = args.batch_size or job_settings.batch_size or DEFAULT_BATCH_SIZE
    try:
        summary = run_message_embedding_pipeline(
            driver,
            provider,
            database=settings.neo4j.database,
            cleanup=args.cleanup,
            dry_run=args.dry_run,
            batch_size=batch_size,
            workers=args.workers,
        )
        logger.info("Message embedding pipeline summary: %s", json.dumps(summary))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
