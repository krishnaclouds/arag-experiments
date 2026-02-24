#!/usr/bin/env bash
# =============================================================================
# run_phase6.sh — A-RAG Phase 6: Full Summarization Evaluation Pipeline
# =============================================================================
#
# Runs the complete pipeline:
#   1.  Prepare ECTSUM dataset (download + chunk + index)
#   2.  Derive FinanceBench QFS tasks (reuses existing index)
#   3–5. Inference: A-RAG / Naive RAG / Stuffing on ECTSUM
#   6–8. Inference: A-RAG / Naive RAG / Stuffing on FinanceBench QFS
#   9–11. Evaluate all ECTSUM systems with G-Eval + ROUGE-2 + retrieval P/R
#  12–14. Evaluate all FinanceBench QFS systems
#  15.   Generate analysis report → results/summarization_report.md
#
# Usage:
#   ./scripts/run_phase6.sh                    # full run (495 ECTSUM items)
#   ./scripts/run_phase6.sh --dev              # dev run (~20 items, quick test)
#   ./scripts/run_phase6.sh --skip-prep        # skip data preparation
#   ./scripts/run_phase6.sh --skip-fbsum       # ECTSUM only, no FinanceBench QFS
#   ./scripts/run_phase6.sh --skip-inference   # eval + report only (preds already exist)
#   ./scripts/run_phase6.sh --workers 5        # override parallel worker count
#
# Prerequisites:
#   uv sync
#   cp .env.example .env  # add ANTHROPIC_API_KEY
#   export TOKENIZERS_PARALLELISM=false
#   export OMP_NUM_THREADS=1
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEV=false
SKIP_PREP=false
SKIP_FBSUM=false
SKIP_INFERENCE=false
WORKERS=3
LIMIT_FLAG=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)             DEV=true ;;
        --skip-prep)       SKIP_PREP=true ;;
        --skip-fbsum)      SKIP_FBSUM=true ;;
        --skip-inference)  SKIP_INFERENCE=true ;;
        --workers)         WORKERS="$2"; shift ;;
        -h|--help)
            grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ "$DEV" == "true" ]]; then
    LIMIT_FLAG="--limit 20"
    echo "=== DEV MODE: processing first 20 items per dataset ==="
fi

# ---------------------------------------------------------------------------
# Step timer helper
# ---------------------------------------------------------------------------
STEP=0
_run() {
    STEP=$((STEP + 1))
    local desc="$1"; shift
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step $STEP: $desc"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local t0=$SECONDS
    "$@"
    echo "  ✓ Done in $((SECONDS - t0))s"
}

# ---------------------------------------------------------------------------
# Step 1 — Prepare ECTSUM
# ---------------------------------------------------------------------------
if [[ "$SKIP_PREP" == "false" ]]; then
    _run "Prepare ECTSUM dataset (download + chunk + index)" \
        uv run python scripts/prepare_ectsum.py $LIMIT_FLAG
else
    echo "(Skipping ECTSUM preparation — --skip-prep)"
fi

# ---------------------------------------------------------------------------
# Step 2 — Derive FinanceBench QFS tasks
# ---------------------------------------------------------------------------
if [[ "$SKIP_FBSUM" == "false" && "$SKIP_PREP" == "false" ]]; then
    _run "Derive FinanceBench QFS tasks (reuses existing index)" \
        uv run python scripts/derive_financebench_sum.py
else
    echo "(Skipping FinanceBench QFS derivation)"
fi

# ---------------------------------------------------------------------------
# Steps 3–5 — Inference on ECTSUM (all three systems)
# ---------------------------------------------------------------------------
if [[ "$SKIP_INFERENCE" == "false" ]]; then
    _run "A-RAG inference on ECTSUM" \
        uv run python scripts/batch_runner.py \
            --config  configs/ectsum.yaml \
            --output  results/ectsum \
            --workers "$WORKERS" $LIMIT_FLAG

    _run "Naive RAG baseline on ECTSUM" \
        uv run python baselines/naive_rag_summary.py \
            --config  configs/ectsum_naive.yaml \
            --output  results/ectsum_naive \
            --workers "$WORKERS" $LIMIT_FLAG

    _run "Long-context stuffing baseline on ECTSUM" \
        uv run python baselines/long_context_summary.py \
            --config  configs/ectsum_stuffing.yaml \
            --output  results/ectsum_stuffing \
            --workers "$WORKERS" $LIMIT_FLAG
else
    echo "(Skipping ECTSUM inference — --skip-inference)"
fi

# ---------------------------------------------------------------------------
# Steps 6–8 — Inference on FinanceBench QFS (all three systems)
# ---------------------------------------------------------------------------
if [[ "$SKIP_FBSUM" == "false" && "$SKIP_INFERENCE" == "false" ]]; then
    FBSUM_QS="data/financebench_sum/questions.json"

    _run "A-RAG inference on FinanceBench QFS" \
        uv run python scripts/batch_runner.py \
            --config    configs/financebench_sum.yaml \
            --questions "$FBSUM_QS" \
            --output    results/financebench_sum \
            --workers   "$WORKERS" $LIMIT_FLAG

    _run "Naive RAG baseline on FinanceBench QFS" \
        uv run python baselines/naive_rag_summary.py \
            --config    configs/financebench_sum.yaml \
            --questions "$FBSUM_QS" \
            --output    results/financebench_sum_naive \
            --workers   "$WORKERS" $LIMIT_FLAG

    _run "Long-context stuffing on FinanceBench QFS" \
        uv run python baselines/long_context_summary.py \
            --config    configs/financebench_sum.yaml \
            --questions "$FBSUM_QS" \
            --output    results/financebench_sum_stuffing \
            --workers   "$WORKERS" $LIMIT_FLAG
fi

# ---------------------------------------------------------------------------
# Steps 9–11 — Evaluate ECTSUM predictions
# ---------------------------------------------------------------------------
GOLD_MAP="data/ectsum/gold_chunk_map.json"
GOLD_FLAG=""
if [[ -f "$GOLD_MAP" ]]; then
    GOLD_FLAG="--gold-chunk-map $GOLD_MAP"
    echo "(gold_chunk_map.json found — retrieval P/R will be computed)"
else
    echo "(gold_chunk_map.json not found — retrieval P/R skipped)"
fi

if [[ -f "results/ectsum/predictions.jsonl" ]]; then
    _run "Evaluate A-RAG on ECTSUM" \
        uv run python scripts/eval.py \
            --predictions results/ectsum/predictions.jsonl \
            --task-type   summarization \
            $GOLD_FLAG \
            --workers "$WORKERS"
fi

if [[ -f "results/ectsum_naive/predictions.jsonl" ]]; then
    _run "Evaluate Naive RAG on ECTSUM" \
        uv run python scripts/eval.py \
            --predictions results/ectsum_naive/predictions.jsonl \
            --task-type   summarization \
            --workers "$WORKERS"
fi

if [[ -f "results/ectsum_stuffing/predictions.jsonl" ]]; then
    _run "Evaluate Long-context stuffing on ECTSUM" \
        uv run python scripts/eval.py \
            --predictions results/ectsum_stuffing/predictions.jsonl \
            --task-type   summarization \
            --workers "$WORKERS"
fi

# ---------------------------------------------------------------------------
# Steps 12–14 — Evaluate FinanceBench QFS predictions
# ---------------------------------------------------------------------------
if [[ "$SKIP_FBSUM" == "false" ]]; then
    if [[ -f "results/financebench_sum/predictions.jsonl" ]]; then
        _run "Evaluate A-RAG on FinanceBench QFS" \
            uv run python scripts/eval.py \
                --predictions results/financebench_sum/predictions.jsonl \
                --task-type   summarization \
                --workers "$WORKERS"
    fi

    if [[ -f "results/financebench_sum_naive/predictions.jsonl" ]]; then
        _run "Evaluate Naive RAG on FinanceBench QFS" \
            uv run python scripts/eval.py \
                --predictions results/financebench_sum_naive/predictions.jsonl \
                --task-type   summarization \
                --workers "$WORKERS"
    fi

    if [[ -f "results/financebench_sum_stuffing/predictions.jsonl" ]]; then
        _run "Evaluate Stuffing on FinanceBench QFS" \
            uv run python scripts/eval.py \
                --predictions results/financebench_sum_stuffing/predictions.jsonl \
                --task-type   summarization \
                --workers "$WORKERS"
    fi
fi

# ---------------------------------------------------------------------------
# Step 15 — Generate analysis report
# ---------------------------------------------------------------------------
FBSUM_ARGS=""
if [[ "$SKIP_FBSUM" == "false" ]]; then
    FBSUM_ARGS="--fbsum-arag results/financebench_sum --fbsum-naive results/financebench_sum_naive --fbsum-stuffing results/financebench_sum_stuffing"
fi

_run "Generate summarization analysis report" \
    uv run python scripts/generate_summarization_report.py \
        --ectsum-arag     results/ectsum \
        --ectsum-naive    results/ectsum_naive \
        --ectsum-stuffing results/ectsum_stuffing \
        $FBSUM_ARGS \
        --output results/summarization_report.md

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 6 complete in ${SECONDS}s."
echo "  Report     → results/summarization_report.md"
echo "  ECTSUM     → results/ectsum{,_naive,_stuffing}/"
if [[ "$SKIP_FBSUM" == "false" ]]; then
    echo "  FB QFS     → results/financebench_sum{,_naive,_stuffing}/"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
