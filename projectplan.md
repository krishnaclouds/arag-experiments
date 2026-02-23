# A-RAG Financial Evaluation — Project Plan

## Objective

Implement and evaluate the **A-RAG** framework ([arXiv:2602.03442](https://arxiv.org/abs/2602.03442)) on financial data. The paper introduces a truly agentic RAG system that exposes three hierarchical retrieval tools to an LLM, letting it autonomously decide retrieval strategy at each step via a ReAct loop, rather than following a fixed pipeline. The goal is to demonstrate that A-RAG outperforms standard single-shot RAG on financial QA tasks that require multi-hop reasoning — a gap that is especially large in financial documents due to dense tables, long filings, and cross-section evidence.

---

## Dataset Selection & Rationale

### Full Dataset Landscape

| Dataset | Size (Qs) | Multi-hop | Free | HuggingFace | License | A-RAG Fit |
|---|---|---|---|---|---|---|
| **FinanceBench** | 150 open / 10,231 full | Moderate | Partial (150) | Yes | MIT (sample) | Real-world SEC filing RAG |
| **DocFinQA** | 7,437 | Yes (long-doc needle) | Yes | No (ACL) | CC BY 4.0 | **Best full-document RAG test** |
| **FinQA** | 8,281 | Yes (arithmetic chain) | Yes | Yes | CC BY 4.0 | Numerical reasoning baseline |
| **MultiHiertt** | ~10,000 | Strong (multi-table) | Yes | No (GitHub) | Likely CC | Multi-table multi-hop strength |
| **ConvFinQA** | 14,115 turns | Yes (cross-turn) | Yes | Yes | MIT | Iterative multi-turn reasoning |
| **TAT-QA** | 16,552 | Partial | Yes | Yes | CC BY 4.0 | Scale, diverse answer types |
| **FinDER** | 5,703 | Yes (compositional) | Yes | Yes | TBD | **Purpose-built RAG eval** |
| **MultiHop-RAG** | 2,556 | Strong (2–4 docs) | Yes | Yes | See repo | Direct multi-hop RAG eval |
| **FinTextQA** | 1,262 | Partial | Yes | Yes | CC BY 4.0 | Long-form generation quality |

---

### Primary: DocFinQA (ACL 2024)

- **Source:** [ACL Anthology](https://aclanthology.org/2024.acl-short.42/) | [arXiv:2401.06915](https://arxiv.org/abs/2401.06915)
- **Size:** 7,437 questions (5,735 train / 780 dev / 922 test) from 801 unique SEC filings
- **Corpus:** Full 10-K filings — average **123,000 words (~150 pages)** per document (vs. 700 words in FinQA). Context is 175× longer than FinQA.
- **Why:** The single best dataset for showcasing A-RAG's advantage. Naive RAG fails catastrophically on 100K+ token documents because a single retrieval step cannot reliably locate evidence buried in financial footnotes, segment breakdowns, and multi-year comparison tables. A-RAG's semantic search → chunk-read iteration is precisely the mechanism needed. Questions come from FinQA so the reasoning programs and execution-accuracy metric carry over directly.
- **License:** CC BY 4.0.

### Primary: FinanceBench (Patronus AI, 2023)

- **Source:** [HuggingFace — PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) | [GitHub](https://github.com/patronus-ai/financebench) | [arXiv:2311.11944](https://arxiv.org/abs/2311.11944)
- **Size:** 150 open questions (from 10,231 total). Full set available by emailing contact@patronus.ai.
- **Corpus:** 360 SEC PDF filings — 10-K (~75%), 10-Q, 8-K, earnings call transcripts from 40 U.S. companies across 9 GICS sectors. ~50,000 total pages.
- **Why:** The industry-standard benchmark for financial RAG pipelines. Directly mirrors AlphaSense's domain. GPT-4 Turbo + naive RAG fails **81% of questions** — the largest baseline gap of any financial QA benchmark. Includes company metadata and page-level evidence annotations enabling retrieval-quality evaluation.
- **Question types:** Numerical calculations (EBITDA, CAGR, YoY change), information retrieval, logical/accounting reasoning, cross-document comparison.
- **License:** MIT (open 150-question sample).

### Secondary: MultiHiertt (ACL 2022)

- **Source:** [GitHub — psunlpgroup/MultiHiertt](https://github.com/psunlpgroup/MultiHiertt) | [ACL Anthology](https://aclanthology.org/2022.acl-long.454/) | [arXiv:2206.01347](https://arxiv.org/abs/2206.01347)
- **Size:** ~10,000 QA pairs from financial reports with 2–5 hierarchical tables each.
- **Why:** The hardest financial multi-hop benchmark. Questions require reasoning chains across multiple nested tables (consolidated vs. segment financials, multi-year comparisons) AND text passages. Standard single-pass retrieval fails because relevant evidence spans table rows in different sections. A-RAG's keyword search can locate table headers; chunk-read retrieves the right rows.
- **License:** Likely CC (ACL 2022 publication — verify on GitHub).

### Tertiary: FinDER (ICLR 2025)

- **Source:** [HuggingFace — Linq-AI-Research/FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER) | [arXiv:2504.15800](https://arxiv.org/abs/2504.15800)
- **Size:** 5,703 query–evidence–answer triplets from 10-K filings. 84.52% qualitative, 15.48% quantitative (49.83% of quantitative require compositional reasoning).
- **Why:** The only dataset built specifically for RAG evaluation in finance, with expert-annotated evidence by an investment banking analyst + CPA. Queries are real professional queries — short, ambiguous, using abbreviations and acronyms. Tests whether A-RAG's autonomous strategy selection is robust under realistic, noisy query conditions.
- **License:** To confirm on HuggingFace.

### Reference: FinQA (EMNLP 2021)

- **Source:** [HuggingFace — ibm-research/finqa](https://huggingface.co/datasets/ibm-research/finqa) | [GitHub](https://github.com/czyssrs/FinQA) | [arXiv:2109.00122](https://arxiv.org/abs/2109.00122)
- **Size:** 8,281 Q&A pairs over 2,800 S&P 500 earnings reports annotated by finance professionals.
- **Why:** The gold standard for multi-hop numerical reasoning. Every question has an annotated symbolic reasoning program (e.g., `divide(subtract(150, 130), 130)`) enabling program-execution accuracy — the strongest possible evaluation since it verifies both the evidence found and the arithmetic chain. Used as the question source for DocFinQA.
- **License:** CC BY 4.0.

### Corpus Source for Custom Experiments

- **`eloukas/edgar-corpus`** (HuggingFace): 10-K filings 1993–2020, pre-split by filing section (Items 1, 1A, 7, 8, etc.). Best for building A-RAG `chunks.json` with clean section boundaries.
- **`PleIAs/SEC`** (HuggingFace): Largest available (245K entries, 1993–2024, 7.2B words). Use for broader coverage.
- **`Bose345/sp500_earnings_transcripts`** (HuggingFace): 33,000+ earnings call transcripts, 685 companies, 2005–2025. Good for a domain beyond annual filings.

---

## Architecture Overview

Following the paper exactly, the system has three layers:

```
Query
  │
  ▼
Agent Loop (ReAct: reason → tool call → observe → repeat, max 15 steps)
  │
  ├── keyword_search(keywords[], top_k)
  │     Exact match score: Σ count(k, chunk) × len(k)
  │     Returns: top-k chunk IDs + keyword-highlighted snippets
  │
  ├── semantic_search(query, top_k)
  │     Dense cosine similarity at sentence level (FAISS)
  │     Returns: top-k chunk IDs + similarity scores
  │
  └── chunk_read(chunk_ids[])
        Full text of requested chunks
        Deduplicates via C_read set (no re-reading)
  │
  ▼
Final Answer
```

**Hierarchical Index (built once per corpus):**
- **Chunk level:** ~1,000-token segments, sentence-boundary aligned
- **Sentence level:** Per-sentence embeddings via `Qwen/Qwen3-Embedding-0.6B` → FAISS index
- **Keyword level:** Runtime exact match (no pre-indexing)

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Python | 3.11+ | Matches reference repo |
| Package manager | `uv` | Reference repo standard |
| LLM (agent) | Claude claude-sonnet-4-6 via Anthropic API | Tool-use support; AlphaSense context |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` | Paper's exact choice |
| Embeddings (CPU fallback) | `sentence-transformers/all-MiniLM-L6-v2` | No GPU required for local dev |
| Vector store | FAISS | Serverless, sufficient for POC corpus size |
| PDF parsing | `pdfplumber` + `pymupdf` | Handles financial PDFs with tables |
| Eval judge LLM | Claude claude-sonnet-4-6 | For LLM-Accuracy metric |
| Demo UI | Streamlit | Fast iteration for demos |

---

## Project Structure

```
ARagPoc/
├── data/
│   ├── financebench/
│   │   ├── pdfs/              # Raw PDF filings (downloaded from FinanceBench)
│   │   ├── chunks.json        # Chunked corpus in A-RAG format ["id:text", ...]
│   │   ├── index/             # FAISS index + sentence metadata
│   │   └── questions.json     # FinanceBench Q&A in A-RAG format
│   ├── docfinqa/
│   │   ├── chunks.json        # Full SEC filing chunks
│   │   ├── index/
│   │   └── questions.json     # FinQA questions mapped to full-doc corpus
│   ├── multihiertt/
│   │   ├── chunks.json
│   │   ├── index/
│   │   └── questions.json
│   └── finder/
│       ├── chunks.json
│       ├── index/
│       └── questions.json
├── src/arag/
│   ├── indexing/
│   │   ├── pdf_parser.py      # PDF → structured text + linearized tables
│   │   ├── chunker.py         # Text → ~1000-token sentence-aligned chunks
│   │   └── build_index.py     # Chunk embeddings → FAISS index
│   ├── tools/
│   │   ├── keyword_search.py  # Exact match scoring per paper formula
│   │   ├── semantic_search.py # FAISS sentence-level cosine search
│   │   └── chunk_read.py      # Full chunk retrieval + C_read dedup tracker
│   ├── agent/
│   │   ├── loop.py            # ReAct agent loop using Anthropic tool-use API
│   │   └── prompts.py         # System prompt + tool schemas
│   └── config.py              # max_loops, top_k, model, embedding model settings
├── scripts/
│   ├── prepare_financebench.py   # Download PDFs, parse, chunk, build index
│   ├── prepare_docfinqa.py       # Download full SEC filings, chunk, build index
│   ├── prepare_multihiertt.py    # Parse multi-table reports → chunks
│   ├── prepare_finder.py         # Download FinDER, convert to A-RAG format
│   ├── batch_runner.py           # Parallel inference over questions.json
│   └── eval.py                   # LLM-Accuracy + Contain-Match + Execution Accuracy
├── baselines/
│   └── naive_rag.py           # Single-shot semantic search baseline for comparison
├── app/
│   └── streamlit_app.py       # Interactive demo UI with reasoning trace
├── results/
│   └── .gitkeep
├── configs/
│   ├── financebench.yaml
│   ├── docfinqa.yaml
│   ├── multihiertt.yaml
│   └── finder.yaml
├── .env.example
├── pyproject.toml
└── CLAUDE.md
```

---

## Implementation Phases

### Phase 0 — Project Setup (Day 1)

1. Initialize `pyproject.toml` with all dependencies (`anthropic`, `faiss-cpu`, `pdfplumber`, `pymupdf`, `sentence-transformers`, `transformers`, `streamlit`, `huggingface_hub`, `datasets`, `tqdm`, `pyyaml`, `ragas`)
2. Create `.env.example` with `ANTHROPIC_API_KEY`
3. Create `configs/*.yaml` files with agent parameters (`max_loops: 15`, `top_k: 5`, `max_token_budget: 128000`)
4. Update `CLAUDE.md` with setup and run commands

### Phase 1 — Core A-RAG Implementation (Days 2–4)

Implement the three tools and agent loop before any dataset-specific work, so they can be tested early.

1. **`keyword_search.py`**: Tokenize keywords, apply paper's scoring formula (`Σ count(k, chunk) × len(k)`), return top-k with highlighted snippets
2. **`semantic_search.py`**: Encode query with embedding model, FAISS search over sentence-level embeddings, aggregate to chunk level, return top-k
3. **`chunk_read.py`**: Return full chunk text; maintain `C_read` set per agent session to prevent redundant token usage
4. **`prompts.py`**: Define Anthropic tool schemas for all three tools + system prompt instructing the agent to gather evidence iteratively and answer when confident
5. **`loop.py`**: Implement ReAct loop using the Anthropic `messages` API with `tools` parameter — stream tool calls from Claude, execute the appropriate tool, append `tool_result` to conversation, continue until Claude emits a text response (final answer) or `max_loops` reached
6. **`build_index.py`**: Chunk embeddings pipeline — given `chunks.json`, produce sentence embeddings and a FAISS index

### Phase 2 — Data Pipeline: DocFinQA (Days 3–5)

**Goal:** Produce `data/docfinqa/chunks.json` and `data/docfinqa/questions.json` in A-RAG format.

DocFinQA is prioritized first because it has the largest A-RAG advantage (naive RAG fails on 100K+ token documents) and shares questions with FinQA so no additional question sourcing is needed.

1. Download DocFinQA questions from ACL Anthology (or reconstruct from FinQA + SEC EDGAR filing references)
2. For each referenced 10-K filing, fetch the full document from SEC EDGAR using the `sec-edgar-downloader` library
3. **Parse and chunk** (`pdf_parser.py` + `chunker.py`):
   - Extract text per section preserving Item labels (Item 1, 1A, 7, 7A, 8, etc.)
   - Detect and linearize tables: convert to `| col1 | col2 | ...` format preserving column headers per row
   - Chunk into ~1,000-token segments at sentence boundaries
   - Prefix each chunk with `[COMPANY: X | FILING: 10-K | SECTION: Item 7]` for keyword-search discoverability
4. Build FAISS index over sentence embeddings
5. Convert FinQA questions to A-RAG format: `qa.question` → `question`, `qa.answer` → `answer`, `qa.gold_inds` → `evidence`, `qa.program` → stored for execution-accuracy eval

### Phase 3 — Data Pipeline: FinanceBench (Days 5–6)

**Goal:** Produce `data/financebench/chunks.json` and `data/financebench/questions.json`.

1. Download 150 questions from HuggingFace (`PatronusAI/financebench`)
2. Download PDFs — FinanceBench metadata includes SEC EDGAR URLs; write a downloader with rate limiting
3. Parse PDFs with `pdfplumber` (better table extraction than pymupdf for financial statements)
4. Chunk and index following the same pipeline as DocFinQA
5. Map FinanceBench `question`/`answer`/`evidence`/`page_number` fields to A-RAG `questions.json` schema

### Phase 4 — Data Pipeline: MultiHiertt & FinDER (Days 6–7)

1. **MultiHiertt**: Clone from GitHub, parse multi-table HTML/JSON report representations, linearize hierarchical tables preserving parent-child header relationships, chunk and index
2. **FinDER**: Download from HuggingFace (`Linq-AI-Research/FinDER`), which already provides `chunks.json`-compatible structure with 10-K passages; convert question/evidence/answer fields to A-RAG format

### Phase 5 — Baseline for Comparison (Day 7)

Implement `baselines/naive_rag.py`: single-shot semantic search (top-5 chunks), concatenate, answer in one LLM call. This is the standard RAG approach A-RAG is compared against. Run on all datasets to establish the baseline gap.

### Phase 6 — Evaluation (Days 8–9)

1. **`batch_runner.py`**: Run A-RAG and naive RAG baseline over all questions with configurable parallelism (`--workers 5`)
2. **`eval.py`**: Three metric tiers:
   - **LLM-Accuracy (primary):** Use Claude as a judge — given `(question, predicted_answer, ground_truth)`, output binary correct/incorrect. Handles "$2.1B" ≡ "2,100 million USD" equivalences that Contain-Match misses. This is the only reliable metric for financial answers.
   - **Contain-Match (secondary):** Substring check. Cheap sanity check; will systematically undercount on financial data due to answer format variation.
   - **Program Execution Accuracy (FinQA/DocFinQA only):** Execute the predicted arithmetic program and compare numerically to gold answer (within ±0.1% tolerance). Strongest metric — verifies both evidence retrieval and reasoning chain correctness.
3. **Retrieval-quality analysis** (where evidence is annotated — FinanceBench, FinDER, FinQA): what fraction of agent's retrieved chunks overlap with gold evidence? Report Recall@K across tools.
4. **RAGAS metrics** (FinDER): Context Recall, Context Precision, Faithfulness using `ragas` library
5. Per-question logging: tool call trace, iterations used, total retrieved tokens, final answer, correctness

### Phase 7 — Demo UI (Days 9–10)

**`app/streamlit_app.py`** showing:
- Query input with example financial questions per dataset
- Dataset/corpus selector (FinanceBench / DocFinQA / MultiHiertt / FinDER)
- Step-by-step reasoning trace: each iteration shows the tool called (name + args) and the observation returned
- Final answer panel with source chunks highlighted
- Side-by-side comparison: A-RAG answer vs naive RAG answer
- Metrics panel: iterations used, tokens retrieved, correctness vs ground truth

---

## Evaluation Plan

### Experiment 1: DocFinQA — Core Long-Document RAG (Primary)

The key experiment. Full 150-page SEC filings vs naive RAG retrieving from the same corpus.

| System | Expected Outcome |
|---|---|
| Naive RAG (top-5 semantic, short snippets) | ~30–40% — fails on deep-buried evidence |
| A-RAG (keyword only) | Improved — precise term matching finds specific line items |
| A-RAG (semantic only) | Improved — dense retrieval finds semantically related sections |
| A-RAG (full — all 3 tools) | Best — iterative retrieval assembles complete arithmetic evidence |

Report: LLM-Accuracy, Contain-Match, Program Execution Accuracy. Analyze average iterations-to-answer and retrieved tokens.

### Experiment 2: FinanceBench — Real-World Financial RAG (Primary)

Run on all 150 open questions. Report LLM-Accuracy and Contain-Match per question category (numerical, retrieval, logical reasoning).

| System | Reported Baseline | Expected with A-RAG |
|---|---|---|
| GPT-4 Turbo + naive RAG | ~19% LLM-Acc | A-RAG target: >45% |
| Claude claude-sonnet-4-6 + naive RAG | TBD | Baseline to beat |
| A-RAG (full — all 3 tools) | — | Best result |

### Experiment 3: MultiHiertt — Multi-Table Reasoning

200-question sample from MultiHiertt test set. Focus on questions requiring evidence from 2+ tables. Report execution accuracy and trace analysis: how many tool calls were needed to locate all relevant tables?

### Experiment 4: FinDER — Realistic Professional Queries

Full FinDER evaluation set. Use RAGAS framework for Context Recall and Faithfulness. This tests robustness on short, ambiguous, real-world queries — the most realistic test of production-readiness.

### Experiment 5: Scaling Analysis (Optional)

Vary `max_loops` (5, 10, 15, 20) on DocFinQA to reproduce the paper's test-time compute scaling result on financial data. Expected: 8–15% accuracy improvement from 5→20 loops.

---

## Evaluation Metrics Reference

| Metric | Datasets | What it measures |
|---|---|---|
| **LLM-Accuracy** | All | Semantic equivalence via Claude judge; handles format variation |
| **Contain-Match** | All | Substring check; cheap but underestimates on financial data |
| **Program Execution Accuracy** | FinQA / DocFinQA | Arithmetic program executed and compared numerically; strongest verification |
| **Context Recall (RAGAS)** | FinDER, FinanceBench | Fraction of gold evidence passages retrieved |
| **Faithfulness (RAGAS)** | FinDER | Claims in answer supported by retrieved context |
| **Recall@K** | FinanceBench, FinQA | Gold evidence chunks in top-K retrieved results |
| **Avg. Iterations / Tokens** | All | Efficiency: how much compute to reach correct answer |

**Key note:** For financial data, LLM-Accuracy must be the primary reported metric. Contain-Match systematically undercounts because financial answers have many equivalent representations ("$2.1B", "2,100 million USD", "approximately 2.1 billion dollars").

---

## Key Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| Financial PDFs with complex table layouts | Use `pdfplumber` for table extraction; linearize tables as `\| col \| col \|` format; preserve column headers per row |
| Hierarchical tables in MultiHiertt (parent/child rows) | Represent parent table headers as prefix on child rows during linearization |
| Financial answer equivalences ($2.1B vs 2,100M) | Rely on LLM-Accuracy as primary metric; add numerical normalization pre-check before Contain-Match |
| PDF download from SEC EDGAR (rate limits, ~10 req/s limit) | Use `sec-edgar-downloader` library; add retry logic and ≥100ms delays between requests |
| GPU requirement for `Qwen3-Embedding-0.6B` | Use `all-MiniLM-L6-v2` as CPU fallback; configurable in `config.py` |
| FinanceBench only has 150 public questions | Use DocFinQA (7,437 questions) as the main statistical dataset; FinanceBench for domain relevance demo |
| Long financial documents exceeding agent token budget | `max_token_budget: 128000` (paper default); C_read dedup mechanism prevents re-reading same chunks |
| DocFinQA full filing access | Cross-reference FinQA `filename` field with SEC EDGAR full-text search to locate original filings |

---

## Data Format Reference

**`chunks.json`** (A-RAG format — same across all datasets):
```json
["0:[COMPANY: Apple Inc. | FILING: 10-K FY2023 | SECTION: Item 7] Apple reported revenue of $383.3 billion...",
 "1:[COMPANY: Apple Inc. | FILING: 10-K FY2023 | SECTION: Item 8] | Year | Revenue | Operating Income |\n| 2023 | 383,285 | 114,301 |\n| 2022 | 394,328 | 119,437 |",
 "2:[COMPANY: Apple Inc. | FILING: 10-K FY2023 | SECTION: Item 7] Products gross margin was 36.5%..."]
```

**`questions.json`** (A-RAG format):
```json
[{
  "id": "docfinqa_0001",
  "source": "docfinqa",
  "question": "What was the year-over-year change in Apple's operating income from FY2022 to FY2023?",
  "answer": "-4.4%",
  "question_type": "numerical",
  "evidence": "Operating income was 114,301 in 2023 vs 119,437 in 2022",
  "evidence_relations": [],
  "program": "subtract(119437, 114301), divide(#0, 119437)"
}]
```

---

## Milestones

| Milestone | Deliverable |
|---|---|
| M1 — Environment ready | `pyproject.toml`, `.env.example`, configs, skeleton modules |
| M2 — A-RAG core running | Three tools + agent loop passing single-question interactive test |
| M3 — DocFinQA pipeline | `chunks.json` + `index/` + `questions.json` for DocFinQA |
| M4 — FinanceBench pipeline | Same for 150 FinanceBench questions |
| M5 — MultiHiertt + FinDER pipelines | Same for secondary datasets |
| M6 — Baseline running | Naive RAG results on all datasets |
| M7 — Full A-RAG evaluation | Results + comparison table across all 4 datasets and all metrics |
| M8 — Demo UI | Streamlit app with live agent trace + side-by-side comparison |

---

## References

- A-RAG paper: [arXiv:2602.03442](https://arxiv.org/abs/2602.03442)
- A-RAG reference implementation: [github.com/Ayanami0730/arag](https://github.com/Ayanami0730/arag)
- A-RAG reference dataset: [huggingface.co/datasets/Ayanami0730/rag_test](https://huggingface.co/datasets/Ayanami0730/rag_test)
- DocFinQA: [arXiv:2401.06915](https://arxiv.org/abs/2401.06915) | [ACL Anthology](https://aclanthology.org/2024.acl-short.42/)
- FinanceBench: [arXiv:2311.11944](https://arxiv.org/abs/2311.11944) | [HuggingFace](https://huggingface.co/datasets/PatronusAI/financebench)
- FinQA: [arXiv:2109.00122](https://arxiv.org/abs/2109.00122) | [HuggingFace](https://huggingface.co/datasets/ibm-research/finqa)
- MultiHiertt: [arXiv:2206.01347](https://arxiv.org/abs/2206.01347) | [GitHub](https://github.com/psunlpgroup/MultiHiertt)
- FinDER: [arXiv:2504.15800](https://arxiv.org/abs/2504.15800) | [HuggingFace](https://huggingface.co/datasets/Linq-AI-Research/FinDER)
- EDGAR Corpus: [HuggingFace — eloukas/edgar-corpus](https://huggingface.co/datasets/eloukas/edgar-corpus)
- Earnings transcripts: [HuggingFace — Bose345/sp500_earnings_transcripts](https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts)
