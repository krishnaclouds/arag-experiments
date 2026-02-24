#!/usr/bin/env python3
"""
Generate the Phase 6 summarization analysis report.

Reads predictions.jsonl + predictions.eval.jsonl from all three systems
(A-RAG, Naive RAG, Long-Context Stuffing) on ECTSUM and optionally
FinanceBench QFS, then writes results/summarization_report.md with:

  - Headline metrics table (faithfulness, coverage, ROUGE-2, retrieval P/R)
  - Cost & latency analysis (P50/P90 latency, cost per coverage point)
  - Error taxonomy breakdown (H/N/O/P/IR/IC/V per system)
  - Retrieval quality analysis (loop count vs recall correlation)
  - Qualitative examples (2 A-RAG wins, 1 stuffing win, 1 failure)
  - Key findings and recommendations

Usage:
    uv run python scripts/generate_summarization_report.py \\
        --ectsum-arag     results/ectsum \\
        --ectsum-naive    results/ectsum_naive \\
        --ectsum-stuffing results/ectsum_stuffing \\
        [--fbsum-arag     results/financebench_sum] \\
        [--fbsum-naive    results/financebench_sum_naive] \\
        [--fbsum-stuffing results/financebench_sum_stuffing] \\
        [--output         results/summarization_report.md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file. Returns empty list if the file doesn't exist."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_system(results_dir: str | None) -> tuple[list[dict], list[dict]]:
    """
    Load (predictions, eval_records) from a results directory.
    Returns ([], []) if the directory or files don't exist.
    """
    if not results_dir:
        return [], []
    d = Path(results_dir)
    preds = _load_jsonl(d / "predictions.jsonl")
    evals = _load_jsonl(d / "predictions.eval.jsonl")
    return preds, evals


def _join(preds: list[dict], evals: list[dict]) -> list[dict]:
    """Merge predictions and eval records on 'id' field."""
    pred_by_id = {p["id"]: p for p in preds}
    result = []
    for e in evals:
        combined = {**pred_by_id.get(e["id"], {}), **e}
        result.append(combined)
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], p: int) -> float:
    """p-th percentile (0–100) of a list, ignoring None values."""
    sv = sorted(v for v in values if v is not None)
    if not sv:
        return 0.0
    idx = max(0, min(int(len(sv) * p / 100), len(sv) - 1))
    return sv[idx]


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 for degenerate inputs."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return round(num / den, 4) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _agg(records: list[dict]) -> dict:
    """Compute aggregate statistics from a list of joined prediction+eval records."""
    if not records:
        return {}
    n = len(records)

    def _vals(key: str) -> list:
        return [r[key] for r in records if r.get(key) is not None]

    faith = _vals("geval_faithfulness")
    cov = _vals("geval_coverage")
    rouge = _vals("rouge2_f1")
    recall = _vals("retrieval_recall")
    precision = _vals("retrieval_precision")
    loops = _vals("loops")
    costs = _vals("cost_usd")
    latencies = _vals("latency_ms")
    words = _vals("word_count")
    bertscore = _vals("bertscore_f1")

    avg_cov = _mean(cov)
    avg_cost = _mean(costs)

    return {
        "n": n,
        "avg_faithfulness": round(_mean(faith), 3),
        "avg_coverage": round(avg_cov, 3),
        "faith_ge4_pct": round(sum(1 for v in faith if v >= 4) / len(faith), 3) if faith else 0.0,
        "cov_ge4_pct": round(sum(1 for v in cov if v >= 4) / len(cov), 3) if cov else 0.0,
        "avg_rouge2": round(_mean(rouge), 4),
        "avg_recall": round(_mean(recall), 4) if recall else None,
        "avg_precision": round(_mean(precision), 4) if precision else None,
        "avg_bertscore": round(_mean(bertscore), 4) if bertscore else None,
        "avg_loops": round(_mean(loops), 1),
        "avg_cost_usd": round(avg_cost, 5),
        "cost_per_cov_pt": round(avg_cost / avg_cov, 5) if avg_cov > 0 else None,
        "p50_latency_s": round(_percentile(latencies, 50) / 1000, 1),
        "p90_latency_s": round(_percentile(latencies, 90) / 1000, 1),
        "avg_words": round(_mean(words), 1),
        "max_loops_pct": round(sum(1 for r in records if r.get("max_loops_reached")) / n, 3),
        "skipped_pct": round(sum(1 for r in records if r.get("skipped")) / n, 3),
    }


def _error_taxonomy(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        for code in r.get("error_codes", []):
            counts[code] = counts.get(code, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Qualitative example selection
# ---------------------------------------------------------------------------

def _select_best(records: list[dict]) -> dict | None:
    """Highest combined faithfulness + coverage, non-skipped."""
    scored = [
        r for r in records
        if r.get("geval_faithfulness") and r.get("geval_coverage")
        and not r.get("skipped")
    ]
    if not scored:
        return None
    return max(scored, key=lambda r: r["geval_faithfulness"] + r["geval_coverage"])


def _select_worst(records: list[dict]) -> dict | None:
    """Lowest combined faithfulness + coverage, non-skipped."""
    scored = [
        r for r in records
        if r.get("geval_faithfulness") and r.get("geval_coverage")
        and not r.get("skipped")
    ]
    if not scored:
        return None
    return min(scored, key=lambda r: r["geval_faithfulness"] + r["geval_coverage"])


def _select_stuffing_win(
    arag_records: list[dict],
    stuffing_records: list[dict],
) -> tuple[dict | None, dict | None]:
    """Find the example with the largest stuffing faithfulness advantage over A-RAG."""
    stuffing_by_id = {r["id"]: r for r in stuffing_records if not r.get("skipped")}
    arag_by_id = {r["id"]: r for r in arag_records}

    best_delta = -999
    best_pair: tuple[dict | None, dict | None] = (None, None)

    for qid, s_rec in stuffing_by_id.items():
        a_rec = arag_by_id.get(qid)
        if not a_rec:
            continue
        delta = s_rec.get("geval_faithfulness", 0) - a_rec.get("geval_faithfulness", 0)
        if delta > best_delta:
            best_delta = delta
            best_pair = (a_rec, s_rec)

    return best_pair if best_delta > 0 else (None, None)


def _select_moderate_arag(records: list[dict], exclude_id: str | None) -> dict | None:
    """A-RAG example with coverage ≥ 4 and 6–15 loops (different from exclude_id)."""
    candidates = [
        r for r in records
        if r.get("geval_coverage", 0) >= 4
        and 6 <= r.get("loops", 0) <= 15
        and r.get("id") != exclude_id
        and not r.get("skipped")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("geval_coverage", 0))


# ---------------------------------------------------------------------------
# Retrieval quality analysis
# ---------------------------------------------------------------------------

def _retrieval_analysis(arag_records: list[dict]) -> dict:
    """Loop-count vs retrieval-recall correlation and loop-bucket breakdown."""
    valid = [
        r for r in arag_records
        if r.get("loops") is not None and r.get("retrieval_recall") is not None
    ]
    if not valid:
        return {}

    loops = [r["loops"] for r in valid]
    recalls = [r["retrieval_recall"] for r in valid]

    corr = _pearson(loops, recalls)

    buckets: dict[str, list[dict]] = {
        "1–5 loops": [r for r in valid if r["loops"] <= 5],
        "6–10 loops": [r for r in valid if 6 <= r["loops"] <= 10],
        "11–15 loops": [r for r in valid if 11 <= r["loops"] <= 15],
        "16–22 loops": [r for r in valid if r["loops"] > 15],
    }

    bucketed = {}
    for name, recs in buckets.items():
        if recs:
            bucketed[name] = {
                "n": len(recs),
                "avg_recall": round(_mean([r["retrieval_recall"] for r in recs]), 3),
                "avg_coverage": round(_mean([r.get("geval_coverage", 0) for r in recs]), 2),
            }

    return {
        "correlation": corr,
        "buckets": bucketed,
        "n_with_recall": len(valid),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _na(v, fmt: str | None = None) -> str:
    if v is None:
        return "N/A"
    if fmt:
        return fmt.format(v)
    return str(v)


def _pct(v) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def _score(v) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def _cost(v) -> str:
    return "N/A" if v is None else f"${v:.4f}"


def _trunc(text: str, max_chars: int = 500) -> str:
    if not text:
        return "(empty)"
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(
    dataset_name: str,
    arag_records: list[dict],
    naive_records: list[dict],
    stuffing_records: list[dict],
    fbsum_section: str = "",
) -> str:
    arag_agg = _agg(arag_records)
    naive_agg = _agg(naive_records)
    stuffing_agg = _agg(stuffing_records)

    today = date.today().isoformat()
    n_a = arag_agg.get("n", 0)
    n_n = naive_agg.get("n", 0)
    n_s = stuffing_agg.get("n", 0)

    arag_cov = arag_agg.get("avg_coverage", 0.0)
    naive_cov = naive_agg.get("avg_coverage", 0.0)
    stuffing_cov = stuffing_agg.get("avg_coverage", 0.0)

    has_recall = arag_agg.get("avg_recall") is not None
    ret = _retrieval_analysis(arag_records)

    error_a = _error_taxonomy(arag_records)
    error_n = _error_taxonomy(naive_records)
    error_s = _error_taxonomy(stuffing_records)
    all_codes = sorted(
        set(error_a) | set(error_n) | set(error_s),
        key=lambda c: -(error_a.get(c, 0) + error_n.get(c, 0) + error_s.get(c, 0)),
    )

    # Qualitative examples
    best = _select_best(arag_records)
    moderate = _select_moderate_arag(arag_records, best["id"] if best else None)
    worst = _select_worst(arag_records)
    a_sw, s_sw = _select_stuffing_win(arag_records, stuffing_records)

    # -----------------------------------------------------------------------
    # Build key-finding paragraph
    # -----------------------------------------------------------------------
    if arag_cov >= stuffing_cov and arag_cov > naive_cov:
        key_finding = (
            f"A-RAG achieves the highest G-Eval Coverage ({arag_cov:.2f}/5.0 vs "
            f"{stuffing_cov:.2f} for stuffing, {naive_cov:.2f} for naive RAG), "
            "demonstrating that iterative retrieval improves synthesis quality "
            "even when the full document fits within the context window."
        )
    elif stuffing_cov > arag_cov:
        key_finding = (
            f"Long-context stuffing achieves higher G-Eval Coverage "
            f"({stuffing_cov:.2f}/5.0) than A-RAG ({arag_cov:.2f}/5.0) on "
            f"{dataset_name}. ECTSUM transcripts are short enough to fit in the "
            "200K context window, so providing the full document without retrieval "
            "overhead benefits coverage on this benchmark. A-RAG's advantage is "
            "expected to emerge on longer documents (e.g., DocFinQA 150-page filings)."
        )
    else:
        key_finding = (
            f"A-RAG and long-context stuffing achieve comparable G-Eval Coverage "
            f"({arag_cov:.2f}/5.0 vs {stuffing_cov:.2f}/5.0) on {dataset_name}. "
            "Both substantially outperform naive RAG "
            f"({naive_cov:.2f}/5.0), confirming that broad document access — "
            "whether via iterative retrieval or direct context stuffing — is "
            "essential for complete summarization."
        )

    # -----------------------------------------------------------------------
    # Assemble markdown
    # -----------------------------------------------------------------------
    L: list[str] = []

    # Header
    L += [
        f"# A-RAG Summarization Evaluation: {dataset_name}",
        "",
        f"**Date:** {today}",
        "**Model:** `claude-sonnet-4-6` (inference) · `claude-haiku-4-5-20251001` (judge)",
        f"**Dataset:** ECTSUM — earnings call transcripts (EMNLP 2022)",
        f"**Systems evaluated:** A-RAG (n={n_a}) · Naive RAG (n={n_n}) · Long-Context Stuffing (n={n_s})",
        "",
        "---",
        "",
    ]

    # ── Section 1: Executive Summary ────────────────────────────────────────
    L += [
        "## 1. Executive Summary",
        "",
        key_finding,
        "",
        "| Metric | A-RAG | Naive RAG | Stuffing |",
        "|---|---|---|---|",
        f"| **G-Eval Coverage** (primary) | **{_score(arag_agg.get('avg_coverage'))}** / 5.0 | {_score(naive_agg.get('avg_coverage'))} / 5.0 | {_score(stuffing_agg.get('avg_coverage'))} / 5.0 |",
        f"| **G-Eval Faithfulness** (primary) | **{_score(arag_agg.get('avg_faithfulness'))}** / 5.0 | {_score(naive_agg.get('avg_faithfulness'))} / 5.0 | {_score(stuffing_agg.get('avg_faithfulness'))} / 5.0 |",
        f"| ROUGE-2 F1 | {_score(arag_agg.get('avg_rouge2'))} | {_score(naive_agg.get('avg_rouge2'))} | {_score(stuffing_agg.get('avg_rouge2'))} |",
    ]
    if has_recall:
        L += [
            f"| Retrieval Recall | {_score(arag_agg.get('avg_recall'))} | — | N/A |",
            f"| Retrieval Precision | {_score(arag_agg.get('avg_precision'))} | — | N/A |",
        ]
    if arag_agg.get("avg_bertscore"):
        L.append(
            f"| BERTScore F1 | {_score(arag_agg.get('avg_bertscore'))} | {_score(naive_agg.get('avg_bertscore'))} | {_score(stuffing_agg.get('avg_bertscore'))} |"
        )
    L += [
        f"| Mean cost / item | {_cost(arag_agg.get('avg_cost_usd'))} | {_cost(naive_agg.get('avg_cost_usd'))} | {_cost(stuffing_agg.get('avg_cost_usd'))} |",
        f"| Avg loops / item | {_na(arag_agg.get('avg_loops'))} | {_na(naive_agg.get('avg_loops'))} | {_na(stuffing_agg.get('avg_loops'))} |",
        "",
        "---",
        "",
    ]

    # ── Section 2: Headline Metrics ─────────────────────────────────────────
    L += [
        "## 2. Headline Metrics",
        "",
        "| Dimension | A-RAG | Naive RAG | Stuffing |",
        "|---|---|---|---|",
        f"| G-Eval Faithfulness (mean) | {_score(arag_agg.get('avg_faithfulness'))} | {_score(naive_agg.get('avg_faithfulness'))} | {_score(stuffing_agg.get('avg_faithfulness'))} |",
        f"| G-Eval Faithfulness (≥ 4) | {_pct(arag_agg.get('faith_ge4_pct'))} | {_pct(naive_agg.get('faith_ge4_pct'))} | {_pct(stuffing_agg.get('faith_ge4_pct'))} |",
        f"| G-Eval Coverage (mean) | {_score(arag_agg.get('avg_coverage'))} | {_score(naive_agg.get('avg_coverage'))} | {_score(stuffing_agg.get('avg_coverage'))} |",
        f"| G-Eval Coverage (≥ 4) | {_pct(arag_agg.get('cov_ge4_pct'))} | {_pct(naive_agg.get('cov_ge4_pct'))} | {_pct(stuffing_agg.get('cov_ge4_pct'))} |",
        f"| ROUGE-2 F1 | {_score(arag_agg.get('avg_rouge2'))} | {_score(naive_agg.get('avg_rouge2'))} | {_score(stuffing_agg.get('avg_rouge2'))} |",
    ]
    if has_recall:
        L += [
            f"| Retrieval Recall | {_score(arag_agg.get('avg_recall'))} | — (no trace) | N/A |",
            f"| Retrieval Precision | {_score(arag_agg.get('avg_precision'))} | — (no trace) | N/A |",
        ]
    if arag_agg.get("avg_bertscore"):
        L.append(
            f"| BERTScore F1 | {_score(arag_agg.get('avg_bertscore'))} | {_score(naive_agg.get('avg_bertscore'))} | {_score(stuffing_agg.get('avg_bertscore'))} |"
        )
    L += [
        f"| Avg word count | {_na(arag_agg.get('avg_words'), '{:.0f}')} | {_na(naive_agg.get('avg_words'), '{:.0f}')} | {_na(stuffing_agg.get('avg_words'), '{:.0f}')} |",
        f"| Max-loops reached | {_pct(arag_agg.get('max_loops_pct'))} | N/A | N/A |",
        f"| Skipped (too large) | N/A | N/A | {_pct(stuffing_agg.get('skipped_pct'))} |",
        f"| Items evaluated | {n_a} | {n_n} | {n_s} |",
        "",
        "---",
        "",
    ]

    # ── Section 3: Cost & Latency ────────────────────────────────────────────
    L += [
        "## 3. Cost & Latency Analysis",
        "",
        "| System | Mean cost/item | P50 latency | P90 latency | Cost per coverage point |",
        "|---|---|---|---|---|",
        f"| A-RAG | {_cost(arag_agg.get('avg_cost_usd'))} | {_na(arag_agg.get('p50_latency_s'), '{:.1f}s')} | {_na(arag_agg.get('p90_latency_s'), '{:.1f}s')} | {_cost(arag_agg.get('cost_per_cov_pt'))} |",
        f"| Naive RAG | {_cost(naive_agg.get('avg_cost_usd'))} | {_na(naive_agg.get('p50_latency_s'), '{:.1f}s')} | {_na(naive_agg.get('p90_latency_s'), '{:.1f}s')} | {_cost(naive_agg.get('cost_per_cov_pt'))} |",
        f"| Long-Context Stuffing | {_cost(stuffing_agg.get('avg_cost_usd'))} | {_na(stuffing_agg.get('p50_latency_s'), '{:.1f}s')} | {_na(stuffing_agg.get('p90_latency_s'), '{:.1f}s')} | {_cost(stuffing_agg.get('cost_per_cov_pt'))} |",
        "",
        "> **Cost per coverage point** = mean cost per item ÷ mean G-Eval Coverage score.",
        "> Lower is more cost-efficient. A-RAG's iterative loops increase cost but may",
        "> deliver higher coverage — whether the overhead is justified depends on the",
        "> coverage gap relative to the cheaper baselines.",
        "",
        "---",
        "",
    ]

    # ── Section 4: Error Taxonomy ────────────────────────────────────────────
    code_desc = {
        "H":  "Hallucination — facts not in retrieved chunks",
        "N":  "Numerical Error — figures retrieved incorrectly",
        "O":  "Omission — key reference facts absent from summary",
        "P":  "Premature Termination — stopped before covering all sections",
        "IR": "Irrelevant Retrieval — retrieved wrong company / period / section",
        "IC": "Incoherence — internally contradictory or broken grammar",
        "V":  "Verbosity — greatly exceeds target length or wrong format",
    }
    n_low_a = sum(1 for r in arag_records if r.get("error_codes"))
    n_low_n = sum(1 for r in naive_records if r.get("error_codes"))
    n_low_s = sum(1 for r in stuffing_records if r.get("error_codes"))

    L += [
        "## 4. Error Taxonomy",
        "",
        "Error codes are assigned to predictions scoring < 3 on faithfulness **or** coverage.",
        f"Low-scoring predictions: A-RAG {n_low_a}/{n_a} · Naive RAG {n_low_n}/{n_n} · Stuffing {n_low_s}/{n_s}",
        "",
    ]
    if all_codes:
        L += [
            "| Code | Description | A-RAG | Naive RAG | Stuffing |",
            "|---|---|---|---|---|",
        ]
        for code in all_codes:
            L.append(
                f"| **{code}** | {code_desc.get(code, code)} | "
                f"{error_a.get(code, 0)} | "
                f"{error_n.get(code, 0)} | "
                f"{error_s.get(code, 0)} |"
            )
        L.append("")
    else:
        L.append("*No error codes assigned — all predictions scored ≥ 3 on both dimensions.*\n")
    L += ["---", ""]

    # ── Section 5: Retrieval Quality Analysis ───────────────────────────────
    L += ["## 5. Retrieval Quality Analysis (A-RAG)", ""]

    if ret:
        corr = ret.get("correlation", 0.0)
        n_ret = ret.get("n_with_recall", 0)
        L += [
            f"Based on {n_ret} items with gold evidence chunk annotations.",
            "",
            f"**Pearson correlation between loop count and retrieval recall: {corr:+.3f}**",
            "",
        ]
        if corr > 0.2:
            L.append(
                "A positive correlation confirms that more loops improve evidence coverage. "
                "Increasing `max_loops` may raise G-Eval Coverage scores."
            )
        elif corr < -0.1:
            L.append(
                "A negative / near-zero correlation suggests the agent reaches recall "
                "saturation quickly. Additional loops add search cost without improving "
                "evidence coverage."
            )
        else:
            L.append(
                "A near-zero correlation suggests recall saturates after the first few "
                "retrieval steps; marginal loop budget adds diminishing returns."
            )
        L.append("")

        buckets = ret.get("buckets", {})
        if buckets:
            L += [
                "| Loop bucket | Items | Avg retrieval recall | Avg G-Eval Coverage |",
                "|---|---|---|---|",
            ]
            for name, bd in buckets.items():
                L.append(
                    f"| {name} | {bd['n']} | {bd['avg_recall']:.3f} | {bd['avg_coverage']:.2f} |"
                )
            L.append("")

        if has_recall:
            avg_r = arag_agg.get("avg_recall", 0)
            L += [
                f"A-RAG average retrieval recall: **{avg_r:.3f}** against gold evidence chunks.  ",
                "(Naive RAG has no retrieval trace; stuffing reads the full document — "
                "recall is implicitly 1.0.)",
                "",
            ]
    else:
        L += [
            "*Retrieval recall data not available — run eval with `--gold-chunk-map` to enable.*",
            "",
        ]
    L += ["---", ""]

    # ── Section 6: Qualitative Examples ─────────────────────────────────────
    L += ["## 6. Qualitative Examples", ""]

    # 6.1 Best A-RAG
    if best:
        faith = best.get("geval_faithfulness", "?")
        cov = best.get("geval_coverage", "?")
        loops = best.get("loops", "?")
        cost_v = best.get("cost_usd")
        L += [
            "### 6.1 A-RAG Win — High Coverage & Faithfulness",
            "",
            f"> **ID:** `{best.get('id', '?')}`  ",
            f"> **Faithfulness:** {faith}/5 · **Coverage:** {cov}/5 · **Loops:** {loops}"
            + (f" · **Cost:** ${cost_v:.4f}" if cost_v else ""),
            ">",
            f"> **Task:** {_trunc(best.get('question', ''), 200)}",
            ">",
            "> **A-RAG summary:**",
            f"> {_trunc(best.get('predicted', ''), 450)}",
            "",
        ]
        if best.get("geval_coverage_reasoning"):
            L += [
                "<details>",
                "<summary>G-Eval Coverage reasoning</summary>",
                "",
                _trunc(best.get("geval_coverage_reasoning", ""), 600),
                "",
                "</details>",
                "",
            ]

    # 6.2 Moderate-loops A-RAG win
    if moderate:
        loops = moderate.get("loops", "?")
        cost_v = moderate.get("cost_usd")
        L += [
            "### 6.2 A-RAG Win — Effective Multi-Section Retrieval",
            "",
            f"> **ID:** `{moderate.get('id', '?')}`  ",
            f"> **Faithfulness:** {moderate.get('geval_faithfulness', '?')}/5 · "
            f"**Coverage:** {moderate.get('geval_coverage', '?')}/5 · **Loops:** {loops}"
            + (f" · **Cost:** ${cost_v:.4f}" if cost_v else ""),
            ">",
            f"> **Task:** {_trunc(moderate.get('question', ''), 200)}",
            ">",
            "> **A-RAG summary:**",
            f"> {_trunc(moderate.get('predicted', ''), 450)}",
            "",
        ]

    # 6.3 Stuffing win
    if s_sw and a_sw:
        L += [
            "### 6.3 Stuffing Win — Full-Context Advantage",
            "",
            f"> **ID:** `{s_sw.get('id', '?')}`",
            ">",
            f"> **A-RAG:** Faith {a_sw.get('geval_faithfulness', '?')}/5 · "
            f"Coverage {a_sw.get('geval_coverage', '?')}/5 · Loops {a_sw.get('loops', '?')}",
            f"> **Stuffing:** Faith {s_sw.get('geval_faithfulness', '?')}/5 · "
            f"Coverage {s_sw.get('geval_coverage', '?')}/5 · Loops 1",
            ">",
            f"> **Task:** {_trunc(s_sw.get('question', ''), 200)}",
            ">",
            "> **Stuffing summary:**",
            f"> {_trunc(s_sw.get('predicted', ''), 450)}",
            "",
        ]
        if s_sw.get("geval_faithfulness_reasoning"):
            L += [
                "<details>",
                "<summary>G-Eval Faithfulness reasoning (stuffing)</summary>",
                "",
                _trunc(s_sw.get("geval_faithfulness_reasoning", ""), 500),
                "",
                "</details>",
                "",
            ]

    # 6.4 A-RAG failure
    if worst:
        faith = worst.get("geval_faithfulness", "?")
        cov = worst.get("geval_coverage", "?")
        loops = worst.get("loops", "?")
        codes = worst.get("error_codes", [])
        L += [
            "### 6.4 A-RAG Failure — Low Coverage with Error Analysis",
            "",
            f"> **ID:** `{worst.get('id', '?')}`  ",
            f"> **Faithfulness:** {faith}/5 · **Coverage:** {cov}/5 · **Loops:** {loops}",
            f"> **Error codes:** {', '.join(codes) if codes else 'none assigned'}",
            ">",
            f"> **Task:** {_trunc(worst.get('question', ''), 200)}",
            ">",
            "> **A-RAG summary:**",
            f"> {_trunc(worst.get('predicted', ''), 450)}",
            "",
        ]
        if worst.get("geval_faithfulness_reasoning"):
            L += [
                "<details>",
                "<summary>G-Eval Faithfulness reasoning</summary>",
                "",
                _trunc(worst.get("geval_faithfulness_reasoning", ""), 600),
                "",
                "</details>",
                "",
            ]

    L += ["---", ""]

    # ── Section 7: Key Findings ──────────────────────────────────────────────
    cov_winner = (
        "A-RAG" if arag_cov >= max(naive_cov, stuffing_cov)
        else "Long-Context Stuffing" if stuffing_cov > naive_cov
        else "Naive RAG"
    )
    arag_faith = arag_agg.get("avg_faithfulness", 0.0)
    stuff_faith = stuffing_agg.get("avg_faithfulness", 0.0)
    naive_faith = naive_agg.get("avg_faithfulness", 0.0)

    cost_ratio_naive = (
        arag_agg.get("avg_cost_usd", 0) / naive_agg.get("avg_cost_usd", 1)
        if naive_agg.get("avg_cost_usd", 0) > 0 else None
    )
    cost_ratio_stuffing = (
        arag_agg.get("avg_cost_usd", 0) / stuffing_agg.get("avg_cost_usd", 1)
        if stuffing_agg.get("avg_cost_usd", 0) > 0 else None
    )

    L += ["## 7. Key Findings", ""]

    finding_1 = (
        f"**Coverage leader: {cov_winner}** — "
        f"A-RAG {arag_cov:.2f}/5 · Stuffing {stuffing_cov:.2f}/5 · Naive RAG {naive_cov:.2f}/5."
    )
    if cov_winner == "A-RAG" and arag_cov - stuffing_cov >= 0.3:
        finding_1 += (
            " Iterative retrieval delivers meaningfully better coverage than "
            "full-document stuffing, suggesting focused search reduces distraction "
            "from irrelevant transcript sections."
        )
    elif cov_winner == "Long-Context Stuffing":
        finding_1 += (
            " Stuffing outperforms iterative retrieval on ECTSUM's short transcripts. "
            "The agent's loop budget is insufficient to cover all key sections "
            "that the model sees at once when the full transcript is in context. "
            "A-RAG's advantage is expected to emerge on longer documents (DocFinQA)."
        )
    else:
        finding_1 += (
            " Both A-RAG and stuffing substantially outperform naive RAG, confirming "
            "that breadth of document access is the key driver of coverage quality."
        )
    L.append(f"1. {finding_1}")
    L.append("")

    if cost_ratio_naive:
        finding_2 = (
            f"**Cost:** A-RAG costs {cost_ratio_naive:.0f}× more than naive RAG"
        )
        if cost_ratio_stuffing:
            finding_2 += f" and {cost_ratio_stuffing:.1f}× more than stuffing."
        else:
            finding_2 += "."
        if arag_cov > stuffing_cov:
            finding_2 += (
                f" The {arag_cov - stuffing_cov:.2f}-point coverage improvement over stuffing "
                f"({_cost(arag_agg.get('cost_per_cov_pt'))} vs "
                f"{_cost(stuffing_agg.get('cost_per_cov_pt'))} cost/pt) "
                "may justify the overhead for high-stakes use cases."
            )
        else:
            finding_2 += (
                " Given stuffing achieves comparable or better coverage at lower cost, "
                "A-RAG's iterative overhead may not be justified for short-document summarization."
            )
        L.append(f"2. {finding_2}")
        L.append("")

    if ret:
        corr = ret.get("correlation", 0.0)
        recall_val = arag_agg.get("avg_recall", 0.0)
        finding_3 = (
            f"**Retrieval recall:** A-RAG average recall is **{recall_val:.3f}** "
            f"against gold evidence chunks (loop/recall correlation: {corr:+.3f})."
        )
        if corr > 0.2:
            finding_3 += " More loops meaningfully improve evidence coverage — increasing `max_loops` may raise Coverage scores."
        elif corr < -0.1:
            finding_3 += " Recall saturates quickly; additional loops do not improve evidence coverage."
        else:
            finding_3 += " Recall is largely independent of loop count, suggesting saturation after the first few steps."
        L.append(f"3. {finding_3}")
        L.append("")

    if all_codes:
        top_a = max(error_a, key=error_a.get, default=None) if error_a else None
        top_n = max(error_n, key=error_n.get, default=None) if error_n else None
        finding_4 = "**Dominant failure modes:**"
        L.append(f"4. {finding_4}")
        if top_a:
            L.append(
                f"   - A-RAG: `{top_a}` ({error_a[top_a]} occurrences) — "
                + code_desc.get(top_a, "")
            )
        if top_n:
            L.append(
                f"   - Naive RAG: `{top_n}` ({error_n[top_n]} occurrences) — "
                + code_desc.get(top_n, "")
            )
        top_s = max(error_s, key=error_s.get, default=None) if error_s else None
        if top_s:
            L.append(
                f"   - Stuffing: `{top_s}` ({error_s[top_s]} occurrences) — "
                + code_desc.get(top_s, "")
            )
        L.append("")

    faith_winner = (
        "A-RAG" if arag_faith >= max(naive_faith, stuff_faith)
        else "Long-Context Stuffing" if stuff_faith >= naive_faith
        else "Naive RAG"
    )
    L.append(
        f"5. **Faithfulness leader: {faith_winner}** — "
        f"A-RAG {arag_faith:.2f}/5 · Stuffing {stuff_faith:.2f}/5 · Naive RAG {naive_faith:.2f}/5. "
        "High faithfulness across all systems reflects that the underlying model is "
        "generally grounded; the main differentiator is coverage (omission), not hallucination."
    )
    L += ["", "---", ""]

    # ── Section 8: Recommendations ───────────────────────────────────────────
    L += [
        "## 8. Recommendations",
        "",
        "| Action | Rationale | Expected Impact |",
        "|---|---|---|",
    ]
    max_loops_pct = arag_agg.get("max_loops_pct", 0.0)
    if max_loops_pct > 0.15:
        L.append(
            f"| Increase `max_loops` 22 → 30 | {max_loops_pct:.1%} of items hit the ceiling | +1–2 pts coverage on constrained items |"
        )
    if cov_winner == "Long-Context Stuffing":
        L.append(
            "| Evaluate A-RAG on DocFinQA (150-page filings) | Stuffing beats A-RAG on short ECTSUM transcripts; A-RAG's advantage should emerge on longer documents | Validates iterative retrieval use case |"
        )
    if error_a.get("O", 0) > error_a.get("H", 0):
        L.append(
            "| Increase `top_k` 5 → 8 | Omission (O) is the dominant A-RAG failure — more candidates per step may cover missed sections | +0.2–0.4 coverage pts |"
        )
    if error_a.get("IR", 0) > 2:
        L.append(
            "| Add metadata filter to semantic search | IR errors indicate off-target retrieval; company/period-aware filtering would improve precision | Reduce IR errors ~50% |"
        )
    L += [
        "| Run BERTScore on all systems | Provides semantic similarity robust to financial paraphrase | Better discriminates stuffing vs A-RAG on paraphrased content |",
        "| Extend to DocFinQA (long 10-K filings) | ECTSUM transcripts fit in context window; A-RAG's advantage is expected to grow on 150-page documents | Stronger business case for iterative retrieval |",
        "",
        "---",
        "",
    ]

    # ── Appendix: FinanceBench QFS ───────────────────────────────────────────
    if fbsum_section:
        L += [fbsum_section, ""]

    # Footer
    L += [
        "---",
        "",
        "*Report generated by ARagPoc Phase 6 evaluation pipeline.*  ",
        "*Raw predictions: `results/ectsum/predictions.jsonl` · `results/ectsum_naive/predictions.jsonl` · `results/ectsum_stuffing/predictions.jsonl`*  ",
        "*Per-item eval: `*.eval.jsonl`*",
    ]

    return "\n".join(L)


# ---------------------------------------------------------------------------
# FinanceBench QFS appendix section
# ---------------------------------------------------------------------------

def _build_fbsum_section(
    arag_records: list[dict],
    naive_records: list[dict],
    stuffing_records: list[dict],
) -> str:
    if not any([arag_records, naive_records, stuffing_records]):
        return ""

    a = _agg(arag_records)
    n = _agg(naive_records)
    s = _agg(stuffing_records)

    L = [
        "## Appendix A: FinanceBench QFS Results",
        "",
        "Query-focused summarization tasks derived from the existing FinanceBench corpus (~80 questions).",
        "Reuses the existing FinanceBench FAISS index — no new data download required.",
        "",
        "| Dimension | A-RAG | Naive RAG | Stuffing |",
        "|---|---|---|---|",
        f"| G-Eval Coverage | {_score(a.get('avg_coverage'))} | {_score(n.get('avg_coverage'))} | {_score(s.get('avg_coverage'))} |",
        f"| G-Eval Faithfulness | {_score(a.get('avg_faithfulness'))} | {_score(n.get('avg_faithfulness'))} | {_score(s.get('avg_faithfulness'))} |",
        f"| ROUGE-2 F1 | {_score(a.get('avg_rouge2'))} | {_score(n.get('avg_rouge2'))} | {_score(s.get('avg_rouge2'))} |",
        f"| Mean cost / item | {_cost(a.get('avg_cost_usd'))} | {_cost(n.get('avg_cost_usd'))} | {_cost(s.get('avg_cost_usd'))} |",
        f"| Avg loops | {_na(a.get('avg_loops'))} | {_na(n.get('avg_loops'))} | {_na(s.get('avg_loops'))} |",
        f"| Items | {a.get('n', '—')} | {n.get('n', '—')} | {s.get('n', '—')} |",
        "",
    ]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Phase 6 summarization analysis report"
    )
    p.add_argument("--ectsum-arag",     default="results/ectsum",          help="Results dir: A-RAG on ECTSUM")
    p.add_argument("--ectsum-naive",    default="results/ectsum_naive",     help="Results dir: Naive RAG on ECTSUM")
    p.add_argument("--ectsum-stuffing", default="results/ectsum_stuffing",  help="Results dir: Stuffing on ECTSUM")
    p.add_argument("--fbsum-arag",      default=None, help="Results dir: A-RAG on FinanceBench QFS (optional)")
    p.add_argument("--fbsum-naive",     default=None, help="Results dir: Naive RAG on FinanceBench QFS (optional)")
    p.add_argument("--fbsum-stuffing",  default=None, help="Results dir: Stuffing on FinanceBench QFS (optional)")
    p.add_argument("--output",          default="results/summarization_report.md", help="Output path for the report")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load ECTSUM systems
    a_preds, a_evals = _load_system(args.ectsum_arag)
    n_preds, n_evals = _load_system(args.ectsum_naive)
    s_preds, s_evals = _load_system(args.ectsum_stuffing)

    a_recs = _join(a_preds, a_evals)
    n_recs = _join(n_preds, n_evals)
    s_recs = _join(s_preds, s_evals)

    print(f"ECTSUM loaded: A-RAG={len(a_recs)} · Naive={len(n_recs)} · Stuffing={len(s_recs)}")

    if not any([a_recs, n_recs, s_recs]):
        print(
            "ERROR: No eval results found. Run the full pipeline first:\n"
            "  ./scripts/run_phase6.sh --dev   (quick test)\n"
            "  ./scripts/run_phase6.sh          (full run)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Load FinanceBench QFS (optional)
    fbsum_sec = ""
    if any([args.fbsum_arag, args.fbsum_naive, args.fbsum_stuffing]):
        fb_a_p, fb_a_e = _load_system(args.fbsum_arag)
        fb_n_p, fb_n_e = _load_system(args.fbsum_naive)
        fb_s_p, fb_s_e = _load_system(args.fbsum_stuffing)
        fb_a = _join(fb_a_p, fb_a_e)
        fb_n = _join(fb_n_p, fb_n_e)
        fb_s = _join(fb_s_p, fb_s_e)
        print(f"FinanceBench QFS loaded: A-RAG={len(fb_a)} · Naive={len(fb_n)} · Stuffing={len(fb_s)}")
        fbsum_sec = _build_fbsum_section(fb_a, fb_n, fb_s)

    report = generate_report(
        dataset_name="ECTSUM",
        arag_records=a_recs,
        naive_records=n_recs,
        stuffing_records=s_recs,
        fbsum_section=fbsum_sec,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
