#!/usr/bin/env python3
"""
Evaluate a predictions.jsonl file produced by batch_runner.py.

Metrics:
  - LLM-Accuracy  : Claude-as-judge semantic equivalence (primary)
  - Contain-Match : ground truth substring in prediction (secondary)

Usage:
    uv run python scripts/eval.py \\
        --predictions results/docfinqa/predictions.jsonl \\
        --workers 5

    # Use a cheaper model for the judge
    uv run python scripts/eval.py \\
        --predictions results/financebench/predictions.jsonl \\
        --judge-model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.config import get_api_key

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating whether a predicted answer correctly answers a financial question.

Question: {question}
Ground Truth: {ground_truth}
Predicted Answer: {predicted}

Is the predicted answer correct? Consider:
- Numerical equivalences: "$2.1B" = "2,100 million" = "approximately 2.1 billion"
- Unit variations: "%" vs "percentage points" vs "basis points"
- Rounding differences within 0.5%
- Different phrasings of the same fact

Respond with ONLY "correct" or "incorrect".
"""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def contain_match(predicted: str, ground_truth: str) -> bool:
    return ground_truth.lower().strip() in predicted.lower()


def llm_judge(
    item: dict,
    client: anthropic.Anthropic,
    model: str,
) -> bool:
    prompt = _JUDGE_PROMPT.format(
        question=item["question"],
        ground_truth=item["ground_truth"],
        predicted=item["predicted"],
    )
    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.content[0].text.strip().lower()
    return verdict.startswith("correct")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate A-RAG predictions")
    p.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    p.add_argument("--workers", type=int, default=5, help="Parallel judge workers")
    p.add_argument(
        "--judge-model",
        default="claude-haiku-4-5-20251001",
        help="Model to use as LLM judge (default: claude-haiku-4-5-20251001)",
    )
    p.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-Accuracy (only compute Contain-Match)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.predictions) as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Evaluating {len(items)} predictions…")

    client = anthropic.Anthropic(api_key=get_api_key())

    cm_correct = 0
    llm_correct = 0
    llm_results: list[bool] = []

    if args.no_llm_judge:
        for item in tqdm(items, desc="Contain-Match"):
            if contain_match(item["predicted"], item["ground_truth"]):
                cm_correct += 1
    else:
        def eval_item(item: dict) -> tuple[bool, bool]:
            cm = contain_match(item["predicted"], item["ground_truth"])
            lm = llm_judge(item, client, args.judge_model)
            return cm, lm

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(eval_item, item): item for item in items}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
                cm, lm = future.result()
                if cm:
                    cm_correct += 1
                if lm:
                    llm_correct += 1
                llm_results.append(lm)

    n = len(items)
    avg_loops = sum(item.get("loops", 0) for item in items) / n
    avg_tokens = (
        sum(item.get("input_tokens", 0) + item.get("output_tokens", 0) for item in items) / n
    )
    max_loops_pct = sum(1 for item in items if item.get("max_loops_reached")) / n

    print(f"\n{'=' * 50}")
    print(f"Results ({n} questions):")
    print(f"  Contain-Match Accuracy : {cm_correct / n:.1%}  ({cm_correct}/{n})")
    if not args.no_llm_judge:
        print(f"  LLM-Accuracy           : {llm_correct / n:.1%}  ({llm_correct}/{n})")
    print(f"  Avg loops / question   : {avg_loops:.1f}")
    print(f"  Avg tokens / question  : {avg_tokens:,.0f}")
    print(f"  Max-loops reached      : {max_loops_pct:.1%}")
    print(f"{'=' * 50}")

    # Write per-question results alongside predictions
    out_path = Path(args.predictions).with_suffix(".eval.jsonl")
    with open(out_path, "w") as f:
        for item, lm in zip(items, llm_results if llm_results else [None] * n):
            f.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "contain_match": contain_match(item["predicted"], item["ground_truth"]),
                        "llm_correct": lm,
                        "loops": item.get("loops"),
                    }
                )
                + "\n"
            )
    print(f"Per-question results → {out_path}")


if __name__ == "__main__":
    main()
