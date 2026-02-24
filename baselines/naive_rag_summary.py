"""
Naive single-shot RAG baseline for summarization.

Retrieves top-k chunks via semantic search, concatenates them, and generates
a summary in a single LLM call — no iteration, no tool use.

Produces the same predictions.jsonl schema as batch_runner.py (including
cost_usd, latency_ms, word_count) for direct comparison with A-RAG.

Usage:
    uv run python baselines/naive_rag_summary.py \\
        --config configs/ectsum.yaml \\
        --output results/ectsum_naive \\
        --workers 3 --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.agent.prompts import SUMMARIZATION_SYSTEM_PROMPT
from src.arag.config import AgentConfig, compute_cost, get_api_key
from src.arag.tools.chunk_read import ChunkReadTool
from src.arag.tools.semantic_search import SemanticSearchTool


def run_naive_rag_summary(
    question: dict,
    config: AgentConfig,
    semantic_tool: SemanticSearchTool,
    chunk_tool: ChunkReadTool,
    client: anthropic.Anthropic,
) -> dict:
    """
    Semantic top-k retrieval followed by a single summarization LLM call.
    chunk_tool.reset() is called before each question for a clean C_read state.
    """
    chunk_tool.reset()
    t_start = time.monotonic()

    search_result = semantic_tool.run(query=question["question"], top_k=config.top_k)
    chunk_ids = [r["chunk_id"] for r in search_result["results"]]
    read_result = chunk_tool.run(chunk_ids=chunk_ids)

    context = "\n\n---\n\n".join(
        f"[Chunk {r['chunk_id']}]\n{r['text']}"
        for r in read_result["results"]
        if "text" in r
    )

    prompt = f"Context:\n{context}\n\nTask: {question['question']}"
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text.strip()
    total_in  = response.usage.input_tokens
    total_out = response.usage.output_tokens

    return {
        "id": question["id"],
        "source": question.get("source", ""),
        "question": question["question"],
        "ground_truth": question["answer"],
        "predicted": answer,
        "loops": 1,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": compute_cost(config.model, total_in, total_out),
        "latency_ms": int((time.monotonic() - t_start) * 1000),
        "word_count": len(answer.split()),
        "max_loops_reached": False,
        "trace": [],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Naive RAG summarization baseline")
    p.add_argument("--config", required=True)
    p.add_argument("--questions", default=None, help="Override questions.json path")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    config  = AgentConfig.from_yaml(args.config)
    api_key = get_api_key()

    questions_path = args.questions or config.chunks_file.replace("chunks.json", "questions.json")
    with open(questions_path) as f:
        questions: list[dict] = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "predictions.jsonl"

    done_ids: set[str] = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])

    remaining = [q for q in questions if q["id"] not in done_ids]
    print(f"Questions: {len(questions)} total | {len(done_ids)} done | {len(remaining)} remaining")
    if not remaining:
        print("Nothing to do.")
        return

    print("Loading tools…")
    semantic_tool = SemanticSearchTool(
        chunks_file=config.chunks_file,
        index_dir=config.index_dir,
        embedding_model=config.embedding_model,
        device=config.embedding_device,
    )
    # One ChunkReadTool per worker so C_read sets are independent
    chunk_tools = [ChunkReadTool(chunks_file=config.chunks_file) for _ in range(args.workers)]
    client = anthropic.Anthropic(api_key=api_key)

    import queue
    tool_pool: queue.Queue = queue.Queue()
    for ct in chunk_tools:
        tool_pool.put(ct)

    def process(question: dict) -> dict:
        ct = tool_pool.get()
        try:
            return run_naive_rag_summary(question, config, semantic_tool, ct, client)
        finally:
            tool_pool.put(ct)

    with open(output_file, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, q): q for q in remaining}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Naive RAG Summary"):
                out_f.write(json.dumps(future.result()) + "\n")
                out_f.flush()

    print(f"\nPredictions written to {output_file}")


if __name__ == "__main__":
    main()
