#!/usr/bin/env python3
"""Generate embeddings for Neo4j facts and populate the vector index."""

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
from memory_agent.embedding_pipeline import run_embedding_pipeline  # noqa: E402
from memory_agent.embeddings import EmbeddingProvider  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("embed_facts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate Neo4j fact embeddings for semantic search.")
    parser.add_argument("--settings-from-env", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--cleanup", action="store_true", default=False, help="Remove embeddings for facts that no longer exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    logger.info("Starting fact embedding pipeline using model %s", settings.embeddings.model)

    provider = EmbeddingProvider(
        model_name=settings.embeddings.model,
        device=settings.embeddings.device,
        cache_dir=settings.embeddings.cache_dir,
    )

    driver_kwargs = {"max_connection_lifetime": settings.neo4j.max_connection_lifetime}
    if settings.neo4j.encrypted:
        driver_kwargs["encrypted"] = True

    driver = GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password),
        **driver_kwargs,
    )
    try:
        summary = run_embedding_pipeline(
            driver,
            provider,
            database=settings.neo4j.database,
            cleanup=args.cleanup,
        )
        logger.info("Embedding pipeline summary: %s", json.dumps(summary))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
