#!/usr/bin/env python3
"""
Evaluate a predictions.jsonl file produced by batch_runner.py.

QA metrics (--task-type qa, the default):
  - LLM-Accuracy  : Claude-as-judge semantic equivalence (primary)
  - Contain-Match : ground truth substring in prediction (secondary)

Summarization metrics (--task-type summarization):
  - G-Eval Faithfulness : chain-of-thought LLM judge, score 1-5
  - G-Eval Coverage     : chain-of-thought LLM judge, score 1-5
  - Retrieval Recall/Precision : chunk overlap vs gold_chunk_map (ECTSUM)
  - ROUGE-2 F1          : n-gram overlap vs reference
  - BERTScore F1        : semantic similarity via finbert (opt-in, --bertscore)
  - Error taxonomy      : H/N/O/P/IR/IC/V codes for low-scoring predictions

Usage:
    # QA evaluation (unchanged)
    uv run python scripts/eval.py \\
        --predictions results/financebench/predictions.jsonl \\
        --judge-model claude-haiku-4-5-20251001

    # Summarization evaluation
    uv run python scripts/eval.py \\
        --predictions results/ectsum/predictions.jsonl \\
        --task-type summarization \\
        --gold-chunk-map data/ectsum/gold_chunk_map.json \\
        --judge-model claude-haiku-4-5-20251001

    # With BERTScore (downloads ProsusAI/finbert ~440MB on first run)
    uv run python scripts/eval.py \\
        --predictions results/ectsum/predictions.jsonl \\
        --task-type summarization --bertscore
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.config import get_api_key
from src.arag.agent.prompts import (
    ERROR_CODE_PROMPT,
    GEVAL_CRITERIA_PROMPT,
    GEVAL_SCORE_PROMPT,
)

# ---------------------------------------------------------------------------
# QA judge prompt (unchanged)
# ---------------------------------------------------------------------------

_QA_JUDGE_PROMPT = """\
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
# QA metrics (unchanged)
# ---------------------------------------------------------------------------

def contain_match(predicted: str, ground_truth: str) -> bool:
    return ground_truth.lower().strip() in predicted.lower()


def llm_judge(item: dict, client: anthropic.Anthropic, model: str) -> bool:
    prompt = _QA_JUDGE_PROMPT.format(
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
# G-Eval helpers
# ---------------------------------------------------------------------------

def _load_criteria_cache(cache_path: Path) -> dict[str, str]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_criteria_cache(cache_path: Path, cache: dict[str, str]) -> None:
    cache_path.write_text(json.dumps(cache, indent=2))


def generate_criteria(
    client: anthropic.Anthropic,
    model: str,
    dimension: str,
    cache: dict[str, str],
    cache_path: Path,
) -> str:
    """Generate (or load from cache) G-Eval criteria for a given dimension."""
    key = f"{model}:{dimension}"
    if key in cache:
        return cache[key]

    prompt = GEVAL_CRITERIA_PROMPT.format(dimension=dimension)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    criteria = response.content[0].text.strip()
    cache[key] = criteria
    _save_criteria_cache(cache_path, cache)
    return criteria


def _parse_geval_score(text: str) -> int | None:
    """Extract the integer score from a G-Eval response. Returns None if not parseable."""
    # Primary: "Final score: N"
    m = re.search(r"[Ff]inal\s+score\s*[:：]\s*([1-5])", text)
    if m:
        return int(m.group(1))
    # Fallback: last standalone digit 1-5 in the text
    matches = re.findall(r"\b([1-5])\b", text)
    if matches:
        return int(matches[-1])
    return None


def _get_retrieved_text(trace: list[dict]) -> str:
    """Concatenate full text of all chunks the agent read during the session."""
    texts = []
    for step in trace:
        if step.get("tool") == "chunk_read":
            for r in step.get("output", {}).get("results", []):
                if "text" in r:
                    texts.append(f"[Chunk {r['chunk_id']}] {r['text']}")
    return "\n\n---\n\n".join(texts) if texts else "(no chunks retrieved)"


def evaluate_geval(
    item: dict,
    client: anthropic.Anthropic,
    model: str,
    criteria: str,
    dimension: str,
) -> tuple[int, str]:
    """Run one G-Eval scoring call. Returns (score 1-5, full reasoning text)."""
    retrieved_chunks = _get_retrieved_text(item.get("trace", []))
    prompt = GEVAL_SCORE_PROMPT.format(
        dimension=dimension,
        criteria=criteria,
        retrieved_chunks=retrieved_chunks[:8000],  # cap to avoid token overflow
        reference=item["ground_truth"],
        predicted=item["predicted"],
    )
    response = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    reasoning = response.content[0].text.strip()
    score = _parse_geval_score(reasoning)
    if score is None:
        score = 3  # default to middle score if parsing fails; logged via reasoning
    return score, reasoning


# ---------------------------------------------------------------------------
# Retrieval precision / recall
# ---------------------------------------------------------------------------

def _extract_read_chunk_ids(trace: list[dict]) -> set[int]:
    """Collect all chunk IDs the agent requested via chunk_read across all loops."""
    ids: set[int] = set()
    for step in trace:
        if step.get("tool") == "chunk_read":
            ids.update(step.get("input", {}).get("chunk_ids", []))
    return ids


def compute_retrieval_pr(
    trace: list[dict],
    gold_chunk_ids: list[int],
) -> tuple[float, float]:
    """
    Compute retrieval recall and precision against gold evidence chunks.

    recall    = |retrieved ∩ gold| / |gold|
    precision = |retrieved ∩ gold| / |retrieved|

    Returns (0.0, 0.0) when gold_chunk_ids is empty or nothing was retrieved.
    """
    if not gold_chunk_ids:
        return 0.0, 0.0
    retrieved = _extract_read_chunk_ids(trace)
    if not retrieved:
        return 0.0, 0.0
    gold_set = set(gold_chunk_ids)
    overlap = retrieved & gold_set
    recall = len(overlap) / len(gold_set)
    precision = len(overlap) / len(retrieved)
    return round(recall, 4), round(precision, 4)


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

_VALID_ERROR_CODES = {"H", "N", "O", "P", "IR", "IC", "V"}


def assign_error_codes(
    client: anthropic.Anthropic,
    model: str,
    faithfulness_reasoning: str,
    coverage_reasoning: str,
) -> list[str]:
    """
    Assign taxonomy error codes based on G-Eval reasoning.
    Only called when faithfulness < 3 or coverage < 3.
    """
    prompt = ERROR_CODE_PROMPT.format(
        faithfulness_reasoning=faithfulness_reasoning,
        coverage_reasoning=coverage_reasoning,
    )
    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip().lower()
    if raw == "none":
        return []
    codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
    return [c for c in codes if c in _VALID_ERROR_CODES]


# ---------------------------------------------------------------------------
# ROUGE-2 and BERTScore
# ---------------------------------------------------------------------------

def rouge2_f1(predicted: str, reference: str) -> float:
    """ROUGE-2 F1. Returns 0.0 if rouge_score is not installed."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
        return round(scorer.score(reference, predicted)["rouge2"].fmeasure, 4)
    except ImportError:
        return 0.0


def bertscore_f1_batch(
    predictions: list[str],
    references: list[str],
    model_type: str = "ProsusAI/finbert",
) -> list[float]:
    """
    BERTScore F1 for a batch of predictions. Returns list of per-item F1 scores.
    Raises ImportError if bert_score is not installed.
    """
    from bert_score import score as _bert_score
    _, _, F1 = _bert_score(
        predictions,
        references,
        model_type=model_type,
        verbose=False,
        batch_size=16,
    )
    return [round(f.item(), 4) for f in F1]


# ---------------------------------------------------------------------------
# Per-item summarization evaluation
# ---------------------------------------------------------------------------

def eval_summarization_item(
    item: dict,
    client: anthropic.Anthropic,
    model: str,
    faith_criteria: str,
    cov_criteria: str,
    gold_chunk_map: dict[str, list[int]],
) -> dict:
    faith_score, faith_reasoning = evaluate_geval(item, client, model, faith_criteria, "faithfulness")
    cov_score, cov_reasoning = evaluate_geval(item, client, model, cov_criteria, "coverage")

    error_codes: list[str] = []
    if faith_score < 3 or cov_score < 3:
        error_codes = assign_error_codes(client, model, faith_reasoning, cov_reasoning)

    gold_ids = gold_chunk_map.get(item["id"], [])
    retrieval_recall, retrieval_precision = compute_retrieval_pr(item.get("trace", []), gold_ids)

    return {
        "id": item["id"],
        "geval_faithfulness": faith_score,
        "geval_faithfulness_reasoning": faith_reasoning,
        "geval_coverage": cov_score,
        "geval_coverage_reasoning": cov_reasoning,
        "retrieval_recall": retrieval_recall,
        "retrieval_precision": retrieval_precision,
        "rouge2_f1": rouge2_f1(item["predicted"], item["ground_truth"]),
        "word_count": item.get("word_count", len(item["predicted"].split())),
        "loops": item.get("loops"),
        "cost_usd": item.get("cost_usd"),
        "latency_ms": item.get("latency_ms"),
        "error_codes": error_codes,
    }


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
        "--task-type",
        default="qa",
        choices=["qa", "summarization"],
        help="Evaluation mode: 'qa' (default) or 'summarization'",
    )
    # QA-only flag
    p.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="(QA only) Skip LLM-Accuracy and only compute Contain-Match",
    )
    # Summarization-only flags
    p.add_argument(
        "--gold-chunk-map",
        default=None,
        help="(Summarization only) Path to gold_chunk_map.json for retrieval P/R",
    )
    p.add_argument(
        "--bertscore",
        action="store_true",
        help="(Summarization only) Compute BERTScore F1 via ProsusAI/finbert (~440MB download)",
    )
    p.add_argument(
        "--regenerate-criteria",
        action="store_true",
        help="(Summarization only) Force regeneration of G-Eval criteria instead of using cache",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# QA evaluation (unchanged logic)
# ---------------------------------------------------------------------------

def run_qa_eval(args: argparse.Namespace) -> None:
    with open(args.predictions) as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Evaluating {len(items)} predictions (QA)…")
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
    avg_tokens = sum(item.get("input_tokens", 0) + item.get("output_tokens", 0) for item in items) / n
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

    out_path = Path(args.predictions).with_suffix(".eval.jsonl")
    with open(out_path, "w") as f:
        for item, lm in zip(items, llm_results if llm_results else [None] * n):
            f.write(
                json.dumps({
                    "id": item["id"],
                    "contain_match": contain_match(item["predicted"], item["ground_truth"]),
                    "llm_correct": lm,
                    "loops": item.get("loops"),
                }) + "\n"
            )
    print(f"Per-question results → {out_path}")


# ---------------------------------------------------------------------------
# Summarization evaluation
# ---------------------------------------------------------------------------

def run_summarization_eval(args: argparse.Namespace) -> None:
    with open(args.predictions) as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Evaluating {len(items)} predictions (summarization)…")
    client = anthropic.Anthropic(api_key=get_api_key())

    # Load gold chunk map if provided
    gold_chunk_map: dict[str, list[int]] = {}
    if args.gold_chunk_map:
        with open(args.gold_chunk_map) as f:
            gold_chunk_map = json.load(f)
        print(f"Gold chunk map loaded: {len(gold_chunk_map)} entries")

    # Load or generate G-Eval criteria (cached per model+dimension)
    cache_path = Path(args.predictions).parent / ".geval_criteria_cache.json"
    cache = {} if args.regenerate_criteria else _load_criteria_cache(cache_path)

    print("Generating/loading G-Eval criteria…")
    faith_criteria = generate_criteria(client, args.judge_model, "faithfulness", cache, cache_path)
    cov_criteria = generate_criteria(client, args.judge_model, "coverage", cache, cache_path)
    print(f"Criteria ready (cache: {cache_path})")

    # Evaluate each item in parallel
    eval_results: list[dict] = [{}] * len(items)

    import logging

    def _eval(idx_item: tuple[int, dict]) -> tuple[int, dict]:
        idx, item = idx_item
        try:
            return idx, eval_summarization_item(
                item, client, args.judge_model,
                faith_criteria, cov_criteria, gold_chunk_map,
            )
        except Exception:
            logging.warning("eval failed for item %s", item.get("id"), exc_info=True)
            return idx, {
                "id": item.get("id"),
                "geval_faithfulness": 3,
                "geval_faithfulness_reasoning": "EVAL_ERROR",
                "geval_coverage": 3,
                "geval_coverage_reasoning": "EVAL_ERROR",
                "retrieval_recall": 0.0,
                "retrieval_precision": 0.0,
                "rouge2_f1": 0.0,
                "word_count": item.get("word_count", len(item.get("predicted", "").split())),
                "loops": item.get("loops"),
                "cost_usd": item.get("cost_usd"),
                "latency_ms": item.get("latency_ms"),
                "error_codes": ["EVAL_ERROR"],
            }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_eval, (i, item)): i for i, item in enumerate(items)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="G-Eval"):
            idx, result = future.result()
            eval_results[idx] = result

    # Optional BERTScore batch (single-threaded, batch mode)
    if args.bertscore:
        print("Computing BERTScore (ProsusAI/finbert)…")
        try:
            scores = bertscore_f1_batch(
                [item["predicted"] for item in items],
                [item["ground_truth"] for item in items],
            )
            for result, score in zip(eval_results, scores):
                result["bertscore_f1"] = score
        except ImportError:
            print("  bert-score not installed — skipping. Run: uv add bert-score")

    # Aggregate summary
    n = len(eval_results)
    avg_faith = sum(r["geval_faithfulness"] for r in eval_results) / n
    avg_cov = sum(r["geval_coverage"] for r in eval_results) / n
    avg_rouge = sum(r["rouge2_f1"] for r in eval_results) / n
    faith_ge4 = sum(1 for r in eval_results if r["geval_faithfulness"] >= 4) / n
    cov_ge4 = sum(1 for r in eval_results if r["geval_coverage"] >= 4) / n

    has_retrieval = any(r["retrieval_recall"] > 0 for r in eval_results)
    avg_recall = sum(r["retrieval_recall"] for r in eval_results) / n if has_retrieval else None
    avg_precision = sum(r["retrieval_precision"] for r in eval_results) / n if has_retrieval else None

    avg_loops = sum(r.get("loops") or 0 for r in eval_results) / n
    avg_cost = sum(r.get("cost_usd") or 0.0 for r in eval_results) / n
    avg_latency = sum(r.get("latency_ms") or 0 for r in eval_results) / n

    # Error taxonomy breakdown
    code_counts: dict[str, int] = {c: 0 for c in sorted(_VALID_ERROR_CODES)}
    low_score_count = 0
    for r in eval_results:
        if r["error_codes"]:
            low_score_count += 1
            for code in r["error_codes"]:
                code_counts[code] = code_counts.get(code, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"Summarization Results ({n} items):")
    print(f"  G-Eval Faithfulness    : {avg_faith:.2f} / 5.0  (≥4: {faith_ge4:.1%})")
    print(f"  G-Eval Coverage        : {avg_cov:.2f} / 5.0  (≥4: {cov_ge4:.1%})")
    print(f"  ROUGE-2 F1             : {avg_rouge:.4f}")
    if args.bertscore and "bertscore_f1" in eval_results[0]:
        avg_bs = sum(r.get("bertscore_f1", 0.0) for r in eval_results) / n
        print(f"  BERTScore F1           : {avg_bs:.4f}")
    if avg_recall is not None:
        print(f"  Retrieval Recall       : {avg_recall:.4f}")
        print(f"  Retrieval Precision    : {avg_precision:.4f}")
    print(f"  Avg loops / item       : {avg_loops:.1f}")
    print(f"  Avg cost / item        : ${avg_cost:.4f}")
    print(f"  Avg latency / item     : {avg_latency / 1000:.1f}s")
    if low_score_count > 0:
        print(f"\n  Error taxonomy ({low_score_count} low-scoring predictions):")
        for code, count in sorted(code_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"    {code:2s} : {count}")
    print(f"{'=' * 60}")

    out_path = Path(args.predictions).with_suffix(".eval.jsonl")
    with open(out_path, "w") as f:
        for result in eval_results:
            f.write(json.dumps(result) + "\n")
    print(f"Per-item results → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.task_type == "summarization":
        run_summarization_eval(args)
    else:
        run_qa_eval(args)


if __name__ == "__main__":
    main()
