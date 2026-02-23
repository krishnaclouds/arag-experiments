#!/usr/bin/env python3
"""
Run A-RAG (or baseline) inference over a questions.json file.

Usage:
    uv run python scripts/batch_runner.py \\
        --config configs/docfinqa.yaml \\
        --output results/docfinqa \\
        --workers 5

    # Limit to first 20 questions for a quick smoke-test
    uv run python scripts/batch_runner.py \\
        --config configs/financebench.yaml \\
        --output results/financebench \\
        --workers 3 --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.agent.loop import AgentLoop
from src.arag.config import AgentConfig, get_api_key
from src.arag.tools.chunk_read import ChunkReadTool
from src.arag.tools.keyword_search import KeywordSearchTool
from src.arag.tools.semantic_search import SemanticSearchTool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A-RAG batch inference runner")
    p.add_argument("--config", required=True, help="Path to dataset config YAML")
    p.add_argument("--questions", help="Override questions.json path from config")
    p.add_argument("--output", required=True, help="Output directory for predictions.jsonl")
    p.add_argument("--workers", type=int, default=5, help="Parallel workers (default 5)")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N questions")
    return p.parse_args()


def _questions_path(config: AgentConfig, override: str | None) -> str:
    if override:
        return override
    # Derive from chunks_file: data/X/chunks.json → data/X/questions.json
    return config.chunks_file.replace("chunks.json", "questions.json")


def _make_agent(config: AgentConfig, api_key: str) -> AgentLoop:
    """Create a fresh AgentLoop with its own tool instances (thread-safe)."""
    return AgentLoop(
        config=config,
        keyword_tool=KeywordSearchTool(chunks_file=config.chunks_file),
        semantic_tool=SemanticSearchTool(
            chunks_file=config.chunks_file,
            index_dir=config.index_dir,
            embedding_model=config.embedding_model,
            device=config.embedding_device,
        ),
        chunk_tool=ChunkReadTool(chunks_file=config.chunks_file),
        api_key=api_key,
    )


def main() -> None:
    args = parse_args()
    config = AgentConfig.from_yaml(args.config)
    api_key = get_api_key()

    questions_path = _questions_path(config, args.questions)
    with open(questions_path) as f:
        questions: list[dict] = json.load(f)

    if args.limit:
        questions = questions[: args.limit]

    # Resume support: skip questions already written to predictions.jsonl
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "predictions.jsonl"

    done_ids: set[str] = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])

    remaining = [q for q in questions if q["id"] not in done_ids]
    print(
        f"Questions: {len(questions)} total | "
        f"{len(done_ids)} already done | {len(remaining)} remaining"
    )
    if not remaining:
        print("Nothing to do.")
        return

    # Pool of agents (one per worker — each has its own C_read set)
    print(f"Loading {args.workers} agent(s)…")
    agent_pool: queue.Queue[AgentLoop] = queue.Queue()
    for _ in range(args.workers):
        agent_pool.put(_make_agent(config, api_key))

    def process(question: dict) -> dict:
        agent = agent_pool.get()
        try:
            result = agent.run(question["question"])
            return {
                "id": question["id"],
                "source": question.get("source", ""),
                "question": question["question"],
                "ground_truth": question["answer"],
                "predicted": result["answer"],
                "loops": result["loops"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "max_loops_reached": result.get("max_loops_reached", False),
                "trace": result["trace"],
            }
        finally:
            agent_pool.put(agent)

    with open(output_file, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, q): q for q in remaining}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running"):
                row = future.result()
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()

    print(f"\nPredictions written to {output_file}")


if __name__ == "__main__":
    main()
