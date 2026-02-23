"""
Naive single-shot RAG baseline for comparison with A-RAG.

Retrieves top-k chunks via semantic search, concatenates them, and
answers in a single LLM call — no iteration, no tool use.

Usage:
    uv run python baselines/naive_rag.py --config configs/financebench.yaml --output results/financebench_naive
    uv run python baselines/naive_rag.py --config configs/financebench.yaml --output results/financebench_naive --limit 20
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.config import AgentConfig, get_api_key
from src.arag.tools.semantic_search import SemanticSearchTool
from src.arag.tools.chunk_read import ChunkReadTool

_SYSTEM = (
    "You are an expert financial analyst. Answer the question using ONLY the provided "
    "context passages. Be precise and include units in numerical answers. "
    "If the answer cannot be found in the context, say so explicitly."
)


def run_naive_rag(
    question: str,
    config: AgentConfig,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Single-shot semantic retrieval → single LLM call.

    Returns the same schema as AgentLoop.run() for easy comparison.
    """
    api_key = api_key or get_api_key()

    semantic_tool = SemanticSearchTool(
        chunks_file=config.chunks_file,
        index_dir=config.index_dir,
        embedding_model=config.embedding_model,
        device=config.embedding_device,
    )
    chunk_tool = ChunkReadTool(chunks_file=config.chunks_file)

    # Retrieve top-k chunks
    search_result = semantic_tool.run(query=question, top_k=config.top_k)
    chunk_ids = [r["chunk_id"] for r in search_result["results"]]
    read_result = chunk_tool.run(chunk_ids=chunk_ids)

    context_parts = [
        f"[Chunk {r['chunk_id']}]\n{r['text']}"
        for r in read_result["results"]
        if "text" in r
    ]
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=config.model,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text.strip()

    return {
        "answer": answer,
        "retrieved_chunk_ids": chunk_ids,
        "loops": 1,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# ---------------------------------------------------------------------------
# Batch CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Naive RAG batch inference")
    p.add_argument("--config", required=True)
    p.add_argument("--questions", help="Override questions.json path")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    config = AgentConfig.from_yaml(args.config)
    api_key = get_api_key()

    questions_path = args.questions or config.chunks_file.replace("chunks.json", "questions.json")
    with open(questions_path) as f:
        questions: list[dict] = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    output_dir = Path(args.output)
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

    # Pre-load tools once — FAISS IndexFlatIP reads are thread-safe
    print("Loading tools…")
    semantic_tool = SemanticSearchTool(
        chunks_file=config.chunks_file,
        index_dir=config.index_dir,
        embedding_model=config.embedding_model,
        device=config.embedding_device,
    )
    chunk_tool = ChunkReadTool(chunks_file=config.chunks_file)
    client = anthropic.Anthropic(api_key=api_key)

    def process(question: dict) -> dict:
        search_result = semantic_tool.run(query=question["question"], top_k=config.top_k)
        chunk_ids = [r["chunk_id"] for r in search_result["results"]]
        read_result = chunk_tool.run(chunk_ids=chunk_ids)
        context = "\n\n---\n\n".join(
            f"[Chunk {r['chunk_id']}]\n{r['text']}"
            for r in read_result["results"]
            if "text" in r
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question['question']}"
        response = client.messages.create(
            model=config.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "id": question["id"],
            "source": question.get("source", ""),
            "question": question["question"],
            "ground_truth": question["answer"],
            "predicted": response.content[0].text.strip(),
            "loops": 1,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "max_loops_reached": False,
        }

    with open(output_file, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, q): q for q in remaining}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Naive RAG"):
                out_f.write(json.dumps(future.result()) + "\n")
                out_f.flush()

    print(f"\nPredictions written to {output_file}")


if __name__ == "__main__":
    main()
