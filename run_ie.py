#!/usr/bin/env python3
"""Run the llama-server information extraction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ie import IEConfig, LlamaServerConfig, run_ie_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the information extraction pass over SQLite data.")
    parser.add_argument("--sqlite", type=Path, default=Path("./discord.db"), help="Path to the SQLite database.")
    parser.add_argument("--window-size", type=int, default=4, help="Message window size to supply to the LLM.")
    parser.add_argument("--max-windows", type=int, help="Optional cap on windows to process (for testing).")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum confidence to store a fact.")
    parser.add_argument("--llama-url", default="http://localhost:8080/v1/chat/completions", help="llama-server chat completions URL.")
    parser.add_argument("--llama-model", default="GLM-4.5-Air", help="Model name to request from llama-server.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate.")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP client timeout (seconds).")
    parser.add_argument("--api-key", help="Optional bearer token if llama-server requires auth.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = IEConfig(
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold,
        max_windows=args.max_windows,
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

    run_ie_job(args.sqlite, config=config, client_config=llama_config)


if __name__ == "__main__":
    main()
