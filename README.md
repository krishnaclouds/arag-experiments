# A-RAG Financial POC

Evaluation framework for **A-RAG** ([arXiv:2602.03442](https://arxiv.org/abs/2602.03442)) on financial QA datasets. A-RAG exposes three hierarchical retrieval tools to an LLM in a ReAct loop, letting it autonomously decide retrieval strategy at each step rather than following a fixed pipeline.

## Results

Evaluated on **FinanceBench** (150 open Q&A pairs over real SEC filings):

| System | LLM-Accuracy | Contain-Match | Avg Loops | Avg Tokens/Q |
|---|---|---|---|---|
| **A-RAG** | **67.3%** | 19.3% | 8.8 | 118,914 |
| Naive RAG | 3.3% | 2.0% | 1 | 991 |

See [`results/financebench_report.md`](results/financebench_report.md) for the full analysis.

## Architecture

```
Question
   │
   ▼
AgentLoop (ReAct)
   ├── 🔑 keyword_search   ← exact match scoring: Σ count(k,chunk) × len(k)
   ├── 🔍 semantic_search  ← FAISS cosine over sentence-level embeddings
   └── 📄 chunk_read       ← full text retrieval with C_read dedup set
   │
   ▼
Final Answer
```

The agent iterates (up to `max_loops=15`) calling tools until it has sufficient evidence, then emits a plain-text answer.

## Datasets

| Dataset | Questions | Description |
|---|---|---|
| **FinanceBench** | 150 | Open Q&A over 84 real SEC PDFs (Patronus AI, 2023) |
| **FinQA** | 1,147 (test) | Numerical reasoning over S&P 500 earnings reports (EMNLP 2021) |
| **FinDER** | 5,703 | Expert-annotated retrieval triplets from 10-K filings (ICLR 2025) |
| **DocFinQA** | ~7,400 | FinQA questions grounded in full 150-page SEC filings |

## Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd ARagPoc
uv sync
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
```

> **Apple Silicon:** The `.env.example` already includes `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1` to prevent a segfault in sentence-transformers on macOS/aarch64.

## Quickstart

### 1. Prepare a dataset

```bash
# FinanceBench (150 questions, downloads PDFs from SEC)
uv run python scripts/prepare_financebench.py

# FinQA (8,281 questions, downloads from GitHub)
uv run python scripts/prepare_finqa.py

# FinDER (5,703 questions, downloads from HuggingFace)
uv run python scripts/prepare_finder.py

# DocFinQA (full SEC 10-K filings via EDGAR)
uv run python scripts/prepare_docfinqa.py --limit 20   # dev run
```

### 2. Run A-RAG inference

```bash
uv run python scripts/batch_runner.py \
  --config configs/financebench.yaml \
  --output results/financebench \
  --workers 3
```

### 3. Evaluate

```bash
# LLM-Accuracy + Contain-Match (Claude Haiku as judge)
uv run python scripts/eval.py \
  --predictions results/financebench/predictions.jsonl \
  --judge-model claude-haiku-4-5-20251001

# Run naive RAG baseline for comparison
uv run python baselines/naive_rag.py \
  --config configs/financebench.yaml \
  --output results/financebench_naive
```

### 4. Interactive demo

```bash
uv run streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### 5. Single-question debug

```bash
uv run python -c "
from src.arag.agent.loop import run_agent
result = run_agent('What is 3M FY2018 capex in USD millions?', dataset='financebench')
print(result['answer'])
print('loops:', result['loops'])
"
```

## Project Structure

```
ARagPoc/
├── src/arag/
│   ├── agent/          # ReAct loop + system prompt
│   ├── tools/          # keyword_search, semantic_search, chunk_read
│   ├── indexing/       # chunker, FAISS index builder, PDF parser
│   └── config.py       # AgentConfig dataclass + YAML loader
├── scripts/
│   ├── prepare_*.py    # Dataset preparation pipelines
│   ├── batch_runner.py # Parallel A-RAG inference
│   └── eval.py         # LLM-Accuracy + Contain-Match evaluation
├── baselines/
│   └── naive_rag.py    # Single-shot RAG baseline
├── configs/            # Per-dataset YAML configs
├── app/
│   └── streamlit_app.py  # Interactive demo UI
├── data/               # Dataset indices (gitignored, rebuild via prepare_*.py)
├── results/            # Evaluation outputs
└── tests/              # Unit tests
```

## Adding a New Dataset

1. Create `scripts/prepare_<dataset>.py` — downloads source, builds `chunks.json` + FAISS index + `questions.json`
2. Add `configs/<dataset>.yaml` with `chunks_file`, `index_dir`, and agent parameters
3. Run `batch_runner.py --config configs/<dataset>.yaml --output results/<dataset>`
4. Run `eval.py --predictions results/<dataset>/predictions.jsonl`

The Streamlit UI auto-detects any dataset that has a complete index.

## Adding a New Tool

1. Create `src/arag/tools/<tool_name>.py` with a class exposing `.name`, `.schema` (Anthropic tool definition dict), and `.run(**kwargs) -> dict`
2. Register it in `src/arag/agent/loop.py` alongside the existing three tools
3. Update `src/arag/agent/prompts.py` to describe the new tool's use case

## Running Tests

```bash
uv run pytest tests/ -v
```

## Configuration Reference

All agent parameters live in `configs/<dataset>.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `model` | `claude-sonnet-4-6` | Inference model |
| `max_loops` | `15` | Max ReAct iterations |
| `top_k` | `5` | Results per tool call |
| `max_token_budget` | `128000` | Token budget guard |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `temperature` | `0.0` | Sampling temperature |
