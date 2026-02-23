# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ARagPoc is a POC evaluation framework for the **A-RAG** agentic retrieval system ([arXiv:2602.03442](https://arxiv.org/abs/2602.03442)) on financial data for AlphaSense. The repo is designed to run controlled experiments across multiple financial QA datasets, retrieval tool variants, and agent configurations, producing head-to-head comparisons against simpler baselines.

**Achieved result:** A-RAG scores **67.3% LLM-Accuracy** vs **3.3% for naive single-shot RAG** on FinanceBench (150 questions over real SEC PDFs).

See `projectplan.md` for the full architecture and evaluation plan. See `results/financebench_report.md` for the detailed evaluation report.

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

# Run A-RAG inference (parallel, resume-safe)
uv run python scripts/batch_runner.py \
  --config configs/financebench.yaml \
  --output results/financebench \
  --workers 3

# Run naive RAG baseline
uv run python baselines/naive_rag.py \
  --config configs/financebench.yaml \
  --output results/financebench_naive \
  --workers 5

# Evaluate predictions (LLM-Accuracy + Contain-Match)
uv run python scripts/eval.py \
  --predictions results/financebench/predictions.jsonl \
  --judge-model claude-haiku-4-5-20251001

# Run tests
uv run pytest tests/ -v

# Launch Streamlit demo
uv run streamlit run app/streamlit_app.py

# Single-question debug
uv run python -c "
from src.arag.agent.loop import run_agent
r = run_agent('What is 3M FY2018 capex in USD millions?', dataset='financebench')
print(r['answer']); print('loops:', r['loops'])
"
```

## Architecture

### Agent Loop (`src/arag/agent/loop.py`)
ReAct-style loop using the Anthropic `messages` API with `tools`. At each step Claude selects one tool, observes the result, then either calls another tool or emits a final answer (`stop_reason == "end_turn"`). Capped at `max_loops` (default 15). If the limit is hit, the agent is prompted for a best-effort answer.

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

### Data Formats
- **`chunks.json`**: `["id:text", ...]` — array of strings, `id` is a zero-based integer prefix.
- **`questions.json`**: `[{id, source, question, answer, question_type, evidence, evidence_relations, program?, exe_ans?}]` — `program` and `exe_ans` are FinQA/DocFinQA-only arithmetic chain fields.
- **`predictions.jsonl`**: One JSON object per line: `{id, question, ground_truth, predicted, loops, input_tokens, output_tokens, max_loops_reached, trace}`.
- **`predictions.eval.jsonl`**: `{id, contain_match, llm_correct, loops}` — judge verdicts alongside predictions.

## Datasets

| Dataset | Script | Config | Questions | Notes |
|---|---|---|---|---|
| **FinanceBench** | `prepare_financebench.py` | `financebench.yaml` | 150 | Primary demo dataset; real SEC PDFs |
| **FinQA** | `prepare_finqa.py` | `finqa.yaml` | 1,147 (test) | Numerical reasoning; corpus from GitHub |
| **FinDER** | `prepare_finder.py` | `finder.yaml` | 5,703 | Expert-annotated RAG triplets |
| **DocFinQA** | `prepare_docfinqa.py` | `docfinqa.yaml` | ~7,400 | Full 150-page SEC filings via EDGAR |

Data files (`chunks.json`, `questions.json`, FAISS indices, PDFs) are **gitignored** — run the prepare script to rebuild.

## Evaluation Metrics

- **LLM-Accuracy** (primary): Claude-as-judge for semantic equivalence. Handles unit variants ($2.1B ≡ 2,100 million), rounding, phrasing differences.
- **Contain-Match** (secondary): Simple substring check. Undercounts badly on narrative answers — treat as a sanity check only.
- **Judge model**: `claude-haiku-4-5-20251001` by default (cheap, fast). Override with `--judge-model`.

## Adding a New Dataset

1. Write `scripts/prepare_<dataset>.py` — produce `data/<dataset>/chunks.json`, `data/<dataset>/index/`, `data/<dataset>/questions.json`.
2. Add `configs/<dataset>.yaml` — copy an existing config and update `chunks_file` and `index_dir`.
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
| `max_loops` | `15` | Higher = more thorough but costlier. 19.3% of FinanceBench questions hit this ceiling. |
| `top_k` | `5` | Results per tool call. Increasing to 8 may recover some ceiling-hit failures. |
| `temperature` | `0.0` | Keep at 0 for reproducible evals. |
| `embedding_model` | `all-MiniLM-L6-v2` | Swap for `Qwen/Qwen3-Embedding-0.6B` for better quality (requires ~2GB RAM). |

## Environment Variables

```
ANTHROPIC_API_KEY=       # required for agent inference and LLM judge
TOKENIZERS_PARALLELISM=false   # prevents segfault on Apple Silicon
OMP_NUM_THREADS=1              # prevents segfault on Apple Silicon
```
