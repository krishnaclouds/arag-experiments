"""
Long-context stuffing baseline for summarization.

No retrieval at all. Loads all chunks for the target document, concatenates
them into a single context, and generates a summary in one LLM call.

This baseline answers the question: does retrieval add value when the full
document fits within the 200K token context window?

If a document is too large (> --token-limit tokens), the prediction is
flagged with skipped=True rather than truncating or raising an error.

Produces the same predictions.jsonl schema as batch_runner.py (including
cost_usd, latency_ms, word_count, skipped) for direct comparison with A-RAG.

Usage:
    uv run python baselines/long_context_summary.py \\
        --config configs/ectsum.yaml \\
        --output results/ectsum_stuffing \\
        --workers 3 --limit 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.agent.prompts import SUMMARIZATION_SYSTEM_PROMPT
from src.arag.config import AgentConfig, compute_cost, get_api_key

# Approximate characters-per-token ratio (conservative)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _load_chunks(chunks_file: str) -> list[tuple[int, str]]:
    """Load chunks.json and return [(chunk_id, text), ...]."""
    with open(chunks_file) as f:
        raw: list[str] = json.load(f)
    result = []
    for entry in raw:
        chunk_id_str, _, text = entry.partition(":")
        result.append((int(chunk_id_str), text))
    return result


def _get_doc_chunks(
    all_chunks: list[tuple[int, str]],
    question: dict,
) -> list[tuple[int, str]]:
    """
    Filter chunks to only those belonging to the document for this question.

    Matching strategy: look for the company name and period in the chunk's
    metadata prefix (case-insensitive).  Falls back to all chunks if no
    company field is present — this means every question gets the full corpus
    context, which is only appropriate for single-document datasets.
    """
    company = (
        question.get("company", "")
        or _extract_from_id(question["id"], "company")
    ).strip().upper()

    period = (
        question.get("period", "")
        or question.get("doc_period", "")
        or _extract_from_id(question["id"], "period")
    ).strip().upper().replace(" ", "_")

    if not company:
        # No document identifier — return all chunks (single-document corpus)
        return all_chunks

    matched = []
    for chunk_id, text in all_chunks:
        text_upper = text.upper()
        if company in text_upper and (not period or period in text_upper):
            matched.append((chunk_id, text))

    return matched if matched else all_chunks  # fallback: full corpus


def _extract_from_id(question_id: str, part: str) -> str:
    """
    Best-effort extraction from IDs like 'ectsum_AAPL_Q3_2019' or
    'fbsum_apple_inc_2023_risks'.
    """
    tokens = re.sub(r"^(ectsum|fbsum)_", "", question_id).split("_")
    if part == "company" and tokens:
        return tokens[0]
    if part == "period" and len(tokens) > 1:
        return "_".join(tokens[1:3])  # e.g. Q3_2019
    return ""


def run_stuffing(
    question: dict,
    config: AgentConfig,
    all_chunks: list[tuple[int, str]],
    client: anthropic.Anthropic,
    token_limit: int,
) -> dict:
    """
    Stuff all document chunks into context and generate a summary in one call.
    Returns a prediction dict with skipped=True if the document is too large.
    """
    t_start = time.monotonic()

    doc_chunks = _get_doc_chunks(all_chunks, question)

    # Build full context string
    context_parts = [f"[Chunk {cid}]\n{text}" for cid, text in doc_chunks]
    full_context  = "\n\n---\n\n".join(context_parts)
    est_tokens    = _estimate_tokens(full_context)

    # Flag and skip if the document is too large for the context window
    if est_tokens > token_limit:
        return {
            "id": question["id"],
            "source": question.get("source", ""),
            "question": question["question"],
            "ground_truth": question["answer"],
            "predicted": "",
            "loops": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0,
            "word_count": 0,
            "max_loops_reached": False,
            "skipped": True,
            "skip_reason": f"Document too large for stuffing ({est_tokens:,} est. tokens > limit {token_limit:,})",
            "doc_chunks_count": len(doc_chunks),
            "trace": [],
        }

    prompt = f"Context:\n{full_context}\n\nTask: {question['question']}"
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    answer    = response.content[0].text.strip()
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
        "skipped": False,
        "doc_chunks_count": len(doc_chunks),
        "trace": [],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Long-context stuffing baseline for summarization")
    p.add_argument("--config", required=True)
    p.add_argument("--questions", default=None, help="Override questions.json path")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--token-limit",
        type=int,
        default=180_000,
        help="Max estimated tokens before flagging a document as too large (default 180000)",
    )
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

    print("Loading chunks…")
    all_chunks = _load_chunks(config.chunks_file)
    print(f"Loaded {len(all_chunks)} chunks from {config.chunks_file}")

    client = anthropic.Anthropic(api_key=api_key)

    def process(question: dict) -> dict:
        return run_stuffing(question, config, all_chunks, client, args.token_limit)

    skipped_count = 0
    with open(output_file, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, q): q for q in remaining}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Stuffing"):
                row = future.result()
                if row.get("skipped"):
                    skipped_count += 1
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()

    total = len(remaining)
    print(f"\nPredictions written to {output_file}")
    print(f"  Processed : {total - skipped_count}/{total}")
    if skipped_count:
        print(f"  Skipped   : {skipped_count} (document too large for {args.token_limit:,}-token window)")


if __name__ == "__main__":
    main()
