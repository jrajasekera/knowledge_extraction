#!/usr/bin/env python3
"""Run the llama-server information extraction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ie import IEConfig, LlamaServerConfig, reset_ie_progress, run_ie_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the information extraction pass over SQLite data.")
    parser.add_argument("--sqlite", type=Path, default=Path("./discord.db"), help="Path to the SQLite database.")
    parser.add_argument("--window-size", type=int, default=8, help="Message window size to supply to the LLM.")
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Process only this many message windows for this invocation; omit when resuming to finish the remainder.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum confidence to store a fact.")
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=3,
        help="Maximum number of simultaneous LLM requests to issue.",
    )
    parser.add_argument("--llama-url", default="http://localhost:8080/v1/chat/completions", help="llama-server chat completions URL.")
    parser.add_argument("--llama-model", default="GLM-4.5-Air", help="Model name to request from llama-server.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens to generate.")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP client timeout (seconds).")
    parser.add_argument("--api-key", help="Optional bearer token if llama-server requires auth.")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent paused IE run.")
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Clear any saved IE progress before starting a new run.",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Clear cached IE window state before running.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run IE even if windows were previously processed and cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = IEConfig(
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold,
        max_windows=args.max_windows,
        max_concurrent_requests=args.max_concurrent_requests,
    )

    llama_config = LlamaServerConfig(
        base_url=args.llama_url,
        model=args.llama_model,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
    )

    if args.reset_progress or args.reset_cache:
        reset_ie_progress(args.sqlite, clear_cache=args.reset_cache)

    stats = run_ie_job(
        args.sqlite,
        config=config,
        client_config=llama_config,
        resume=args.resume,
        use_cache=not args.force,
    )

    summary = stats.as_dict()
    chunk_text = ""
    target = summary.get("target_windows")
    if isinstance(target, int) and target > 0:
        chunk_text = f" chunk_target={target}"
    print(
        f"[IE] Summary: run_id={summary['run_id']} processed={summary['processed_windows']}"
        f" skipped={summary['skipped_windows']} cached={summary.get('cached_windows', 0)}"
        f" total_processed={summary['total_processed']}"
        f"/{summary['total_windows']} completed={summary['completed']}{chunk_text}"
    )


if __name__ == "__main__":
    main()
