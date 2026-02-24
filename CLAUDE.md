# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ARagPoc is a POC evaluation framework for the **A-RAG** agentic retrieval system ([arXiv:2602.03442](https://arxiv.org/abs/2602.03442)) on financial data for AlphaSense. The repo is designed to run controlled experiments across multiple financial QA datasets, retrieval tool variants, and agent configurations, producing head-to-head comparisons against simpler baselines.

**Achieved result:** A-RAG scores **67.3% LLM-Accuracy** vs **3.3% for naive single-shot RAG** on FinanceBench (150 questions over real SEC PDFs).

The framework is being extended to support **summarization tasks** — see `summarization_plan.md` for the full design.

See `projectplan.md` for the QA architecture and dataset plan. See `results/financebench_report.md` for the detailed QA evaluation report.

## Setup

```bash
uv sync                  # install all dependencies (including pytest)
cp .env.example .env     # fill in ANTHROPIC_API_KEY
```

> **Apple Silicon:** `.env.example` already contains `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1`. These must be set (or exported) before running any script that loads sentence-transformers, or a segfault will occur during FAISS index building.

## Common Commands

```bash
# Prepare a dataset (downloads data, builds chunks.json + FAISS index + questions.json)
uv run python scripts/prepare_financebench.py
uv run python scripts/prepare_finqa.py
uv run python scripts/prepare_finder.py
uv run python scripts/prepare_docfinqa.py --limit 20   # dev run
uv run python scripts/prepare_ectsum.py --limit 20     # summarization dev run (Phase 3)

# Derive FinanceBench query-focused summarization tasks (reuses existing index)
uv run python scripts/derive_financebench_sum.py

# Run A-RAG inference (parallel, resume-safe)
uv run python scripts/batch_runner.py \
  --config configs/financebench.yaml \
  --output results/financebench \
  --workers 3

# Run A-RAG on summarization datasets
uv run python scripts/batch_runner.py \
  --config configs/ectsum.yaml \
  --output results/ectsum \
  --workers 3 --limit 20

# Run naive RAG baseline (QA)
uv run python baselines/naive_rag.py \
  --config configs/financebench.yaml \
  --output results/financebench_naive \
  --workers 5

# Run naive RAG baseline (summarization)
uv run python baselines/naive_rag_summary.py \
  --config configs/ectsum.yaml \
  --output results/ectsum_naive \
  --workers 3

# Run long-context stuffing baseline (summarization only)
uv run python baselines/long_context_summary.py \
  --config configs/ectsum.yaml \
  --output results/ectsum_stuffing \
  --workers 3

# Evaluate QA predictions (LLM-Accuracy + Contain-Match)
uv run python scripts/eval.py \
  --predictions results/financebench/predictions.jsonl \
  --judge-model claude-haiku-4-5-20251001

# Evaluate summarization predictions (G-Eval + ROUGE-2 + retrieval P/R)
uv run python scripts/eval.py \
  --predictions results/ectsum/predictions.jsonl \
  --task-type summarization \
  --gold-chunk-map data/ectsum/gold_chunk_map.json \
  --judge-model claude-haiku-4-5-20251001

# Evaluate with BERTScore (slow — downloads ProsusAI/finbert ~440MB on first run)
uv run python scripts/eval.py \
  --predictions results/ectsum/predictions.jsonl \
  --task-type summarization \
  --bertscore

# Run tests
uv run pytest tests/ -v

# Launch Streamlit demo
uv run streamlit run app/streamlit_app.py

# Single-question debug (QA)
uv run python -c "
from src.arag.agent.loop import run_agent
r = run_agent('What is 3M FY2018 capex in USD millions?', dataset='financebench')
print(r['answer']); print('loops:', r['loops']); print('cost:', r['cost_usd'])
"
```

## Architecture

### Agent Loop (`src/arag/agent/loop.py`)
ReAct-style loop using the Anthropic `messages` API with `tools`. At each step Claude selects one tool, observes the result, then either calls another tool or emits a final answer (`stop_reason == "end_turn"`). Capped at `max_loops` (default 15 for QA, 22 for summarization). If the limit is hit, the agent is prompted for a best-effort answer.

The system prompt is selected by `config.task_type`: `"qa"` uses `SYSTEM_PROMPT`, `"summarization"` uses `SUMMARIZATION_SYSTEM_PROMPT`. Every result dict now includes `cost_usd`, `latency_ms`, and `word_count`.

### Three Retrieval Tools (`src/arag/tools/`)
- **`keyword_search.py`** — Exact-match scoring: `Σ count(k, chunk) × len(k)`. Returns top-k chunks with highlighted snippets. The paper formula: longer keywords score higher, rewarding precision.
- **`semantic_search.py`** — FAISS `IndexFlatIP` cosine similarity over sentence-level embeddings. Query is embedded at runtime; returns top-k chunks after max-pooling sentence scores to chunk level.
- **`chunk_read.py`** — Returns full chunk text. Maintains a `C_read: set[int]` per session; already-read chunk IDs are silently skipped, preventing the agent from re-reading the same context.

### Hierarchical Index (`src/arag/indexing/`)
- **Chunks** (`chunks.json`): `["id:metadata text", ...]` — one entry per 1000-token segment, sentence-boundary aligned. Metadata prefix: `[COMPANY: X | FILING: Y | SECTION: Z]`.
- **Sentence index** (`index/sentences.faiss` + `index/sentence_map.json`): Each sentence is embedded independently; FAISS stores all sentence vectors. `sentence_map.json` maps sentence index → chunk_id for look-up after search.
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, fast CPU inference). Can be swapped in any config YAML.

### Config (`src/arag/config.py`)
`AgentConfig` dataclass loaded from YAML. `from_yaml("configs/financebench.yaml")` is the standard entry point. `get_api_key()` reads from `ANTHROPIC_API_KEY` env var (`.env` is loaded automatically via `python-dotenv`).

`PRICING` dict and `compute_cost(model, input_tokens, output_tokens)` are module-level — used by the agent loop and available to eval scripts.

### Data Formats
- **`chunks.json`**: `["id:text", ...]` — array of strings, `id` is a zero-based integer prefix.
- **`questions.json`**: `[{id, source, question, answer, question_type, evidence, evidence_relations, program?, exe_ans?, summarization_style?}]` — `program` and `exe_ans` are FinQA/DocFinQA-only; `summarization_style` is ECTSUM/summarization-only.
- **`predictions.jsonl`**: One JSON object per line:
  ```json
  {
    "id", "source", "question", "ground_truth", "predicted",
    "loops", "input_tokens", "output_tokens",
    "cost_usd",       ← estimated API cost in USD
    "latency_ms",     ← wall-clock time from first call to answer
    "word_count",     ← word count of predicted answer/summary
    "max_loops_reached", "trace"
  }
  ```
- **`predictions.eval.jsonl`** (QA): `{id, contain_match, llm_correct, loops}`.
- **`predictions.eval.jsonl`** (summarization): `{id, geval_faithfulness, geval_faithfulness_reasoning, geval_coverage, geval_coverage_reasoning, retrieval_recall, retrieval_precision, rouge2_f1, bertscore_f1?, word_count, loops, cost_usd, latency_ms, error_codes}`.
- **`gold_chunk_map.json`** (ECTSUM only): `{question_id: [chunk_id, ...]}` — maps each question to the chunks containing ECTSUM's expert-extracted key factual sentences. Used for retrieval P/R evaluation.

## Datasets

### QA Datasets

| Dataset | Script | Config | Questions | Notes |
|---|---|---|---|---|
| **FinanceBench** | `prepare_financebench.py` | `financebench.yaml` | 150 | Primary demo dataset; real SEC PDFs |
| **FinQA** | `prepare_finqa.py` | `finqa.yaml` | 1,147 (test) | Numerical reasoning; corpus from GitHub |
| **FinDER** | `prepare_finder.py` | `finder.yaml` | 5,703 | Expert-annotated RAG triplets |
| **DocFinQA** | `prepare_docfinqa.py` | `docfinqa.yaml` | ~7,400 | Full 150-page SEC filings via EDGAR |

### Summarization Datasets

| Dataset | Script | Config | Items | Notes |
|---|---|---|---|---|
| **ECTSUM** | `prepare_ectsum.py` | `ectsum.yaml` | 495 | Earnings call transcripts; expert summaries + key sentences for retrieval P/R |
| **FinanceBench QFS** | `derive_financebench_sum.py` | `financebench_sum.yaml` | ~80 | Derived query-focused tasks; reuses existing FinanceBench index |

Data files (`chunks.json`, `questions.json`, FAISS indices, PDFs) are **gitignored** — run the prepare script to rebuild.

## Evaluation Metrics

### QA Metrics
- **LLM-Accuracy** (primary): Claude-as-judge for semantic equivalence. Handles unit variants ($2.1B ≡ 2,100 million), rounding, phrasing differences.
- **Contain-Match** (secondary): Simple substring check. Undercounts badly on narrative answers — treat as a sanity check only.
- **Judge model**: `claude-haiku-4-5-20251001` by default (cheap, fast). Override with `--judge-model`.

### Summarization Metrics
- **G-Eval Faithfulness** (primary): Chain-of-thought LLM judge. First generates 5 named criteria, then reasons through each before emitting a score 1–5. Targets hallucination. Criteria are cached in `{results_dir}/.geval_criteria_cache.json`.
- **G-Eval Coverage** (primary): Same G-Eval protocol. Targets omission of key facts — score 1–5.
- **Error Taxonomy** (diagnostic): For any prediction scoring < 3 on either dimension, error codes are assigned: H (hallucination), N (numerical error), O (omission), P (premature termination), IR (irrelevant retrieval), IC (incoherence), V (verbosity).
- **Retrieval Recall / Precision** (primary for retrieval quality): Requires `gold_chunk_map.json` (ECTSUM only). Measures whether the agent's `chunk_read` calls covered the gold-evidence chunks.
- **ROUGE-2 F1** (secondary): N-gram overlap vs reference. Treat as lower bound.
- **BERTScore F1** (secondary, opt-in): Semantic similarity via `ProsusAI/finbert`. Enable with `--bertscore`.

### Three-System Comparison (summarization)
Every summarization dataset is evaluated against three systems to isolate where A-RAG adds value:
1. **A-RAG** — full ReAct agent, up to 22 loops
2. **Naive RAG** (`baselines/naive_rag_summary.py`) — top-k semantic search → single LLM call
3. **Long-Context Stuffing** (`baselines/long_context_summary.py`) — full document in context, no retrieval

## Adding a New Dataset

1. Write `scripts/prepare_<dataset>.py` — produce `data/<dataset>/chunks.json`, `data/<dataset>/index/`, `data/<dataset>/questions.json`.
2. Add `configs/<dataset>.yaml` — copy an existing config and update `chunks_file`, `index_dir`, and `task_type`.
3. Add a `.gitkeep` in `data/<dataset>/`.
4. The Streamlit UI auto-discovers any dataset with a complete index.

## Adding a New Tool

1. Create `src/arag/tools/<tool>.py` — expose `.name: str`, `.schema: dict` (Anthropic tool definition), `.run(**kwargs) -> dict`.
2. Register it in `AgentLoop.__init__` in `src/arag/agent/loop.py`.
3. Update `src/arag/agent/prompts.py` to describe when to use the new tool.
4. Add unit tests in `tests/test_tools.py`.

## Key Config Parameters

| Parameter | Default | Effect |
|---|---|---|
| `max_loops` | `15` (QA) / `22` (summarization) | Higher = more thorough but costlier. 19.3% of FinanceBench QA questions hit the ceiling. |
| `top_k` | `5` | Results per tool call. Increasing to 8 may recover some ceiling-hit failures. |
| `temperature` | `0.0` | Keep at 0 for reproducible evals. |
| `embedding_model` | `all-MiniLM-L6-v2` | Swap for `Qwen/Qwen3-Embedding-0.6B` for better quality (requires ~2GB RAM). |
| `task_type` | `"qa"` | `"qa"` selects QA system prompt; `"summarization"` selects the broad-retrieval summarization prompt. |
| `summarization_style` | `"earnings_call"` | Metadata only — used in questions.json and reporting. `earnings_call` \| `section` \| `query_focused` \| `comparison`. |
| `target_length` | `200` | Approximate target word count communicated to the summarization prompt. |
| `max_token_budget` | `128000` (QA) / `200000` (summarization) | Global token guard per question. |

## Environment Variables

```
ANTHROPIC_API_KEY=       # required for agent inference and LLM judge
TOKENIZERS_PARALLELISM=false   # prevents segfault on Apple Silicon
OMP_NUM_THREADS=1              # prevents segfault on Apple Silicon
```

## Summarization Implementation Status

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Config fields, prompts, cost/latency tracking | ✅ Done |
| **Phase 2** | G-Eval + error taxonomy in `eval.py` | ✅ Done |
| **Phase 3** | ECTSUM dataset pipeline (`prepare_ectsum.py`) | ✅ Done |
| **Phase 4** | FinanceBench derived QFS tasks | ✅ Done |
| **Phase 5** | Three-baseline setup (naive RAG + long-context stuffing) | ✅ Done |
| **Phase 6** | Full evaluation run and analysis report | Pending |
| **Phase 7** | Streamlit UI extensions | Pending |
