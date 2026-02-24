# A-RAG Summarization Evaluation — Project Plan

## Objective

Extend the ARagPoc evaluation framework to support **summarization tasks** alongside the existing Q&A evaluation. Summarization is a fundamentally different task: rather than extracting a specific fact from a document, the system must synthesize a coherent, faithful, and complete summary from evidence spread across multiple chunks.

This plan covers the full stack: new datasets, a new agent mode, new evaluation metrics, three baselines (including a long-context stuffing baseline), retrieval quality measurement, cost/latency tracking, and an upfront error taxonomy — all designed to slot into the existing A-RAG infrastructure with minimal disruption to the QA pipeline.

---

## Why Summarization Differs from QA

| Dimension | Q&A (current) | Summarization (new) |
|---|---|---|
| **Task** | Extract a specific fact or number | Synthesize a multi-sentence narrative |
| **Ground truth** | Single canonical answer | Reference summary (one of many valid outputs) |
| **Agent goal** | Retrieve until confident; answer concisely | Retrieve broadly; synthesize coherently |
| **Output length** | 1–3 sentences | 100–500 words (varies by task type) |
| **Evaluation** | Binary correct/incorrect (LLM-judge) | Multi-dimensional: faithfulness, coverage, coherence, conciseness |
| **Tool use pattern** | Targeted: keyword → narrow chunk_read | Broad: semantic → many chunk_reads, then synthesize |
| **Max loops** | 15 (enough for fact retrieval) | 20–25 (must cover multiple document sections) |
| **Failure mode** | Hallucination, wrong number | Omission of key facts, fabrication, incoherence |

---

## Summarization Task Types (Financial Domain)

### Type 1: Document Section Summary
Summarize a specific section of a 10-K (e.g., Risk Factors, MD&A, Business Overview). The most directly useful for AlphaSense.
- Input: `"Summarize the key risk factors from Apple's 2023 10-K"`
- Output: 200–400 word narrative covering the top risks
- Ground truth: Analyst-written reference summaries or LLM-generated references with human verification

### Type 2: Earnings Call Summary
Distill an earnings call transcript into management commentary, key metrics, and guidance.
- Input: `"Summarize the Q3 2023 earnings call for Microsoft"`
- Output: 150–300 words covering: headline numbers, management tone, outlook
- Ground truth: ECTSUM dataset provides analyst-written reference summaries for 495 calls

### Type 3: Query-Focused Summary (QFS)
Given a specific topic or investment angle, retrieve and synthesize relevant passages.
- Input: `"Summarize how rising interest rates affected Goldman Sachs in FY2023"`
- Output: 200–350 words synthesizing relevant sections across the filing
- Ground truth: Derived from existing QA datasets with multi-part answers, or hand-curated

### Type 4: Year-over-Year Change Summary
Compare two filings and summarize what changed materially.
- Input: `"What changed materially for 3M between FY2022 and FY2023?"`
- Output: Structured narrative covering revenue, margins, segment performance, risks
- Ground truth: Analyst reports or LLM-generated comparison summaries

---

## Error Taxonomy

Defined upfront so failures are consistently categorised at eval time — critical for knowing which component to fix. Every failed or low-scoring prediction is tagged with one or more of these during analysis.

| Code | Name | Definition | Typical cause |
|---|---|---|---|
| **H** | Hallucination | Summary asserts a fact not present in any retrieved chunk | Model generates plausible-but-unsupported numbers or events |
| **N** | Numerical Error | A figure is retrieved but transcribed or computed incorrectly (e.g., "$1.2B" → "$1.2M") | Unit confusion, rounding in reasoning |
| **O** | Omission | A key fact present in the reference summary is absent from the predicted summary | Coverage failure; agent stopped retrieving too early |
| **P** | Premature Termination | Agent emits a final answer after fewer than expected loops without covering all relevant sections | Prompt not emphatic enough about breadth requirement |
| **IR** | Irrelevant Retrieval | Agent retrieves chunks from the wrong company, period, or section, consuming loop budget without useful content | Keyword/semantic search returning off-target results |
| **IC** | Incoherence | Summary is internally contradictory or grammatically broken | Temperature > 0 or context overflow |
| **V** | Verbosity / Off-format | Summary greatly exceeds target length or fails to follow the required format | System prompt formatting guidance ignored |

**How taxonomy is used**:
- `eval.py` adds an `error_codes: list[str]` field to `predictions.eval.jsonl` for any prediction with faithfulness < 3 or coverage < 3
- Codes are assigned by a dedicated G-Eval taxonomy step (see Evaluation Metrics below)
- The analysis report aggregates error code frequency per system (A-RAG vs each baseline) to identify the dominant failure mode

---

## Three-System Comparison

Rather than a single naive RAG baseline, this plan evaluates three systems. This isolates where A-RAG's value comes from.

| System | Description | Key question it answers |
|---|---|---|
| **A-RAG** | Full ReAct agent with 3 tools, up to 22 loops | Ceiling — what does iterative retrieval achieve? |
| **Naive RAG** | Semantic top-k retrieval → single LLM call | Does retrieval + one-shot generation beat nothing? |
| **Long-Context Stuffing** | Full document in context, no retrieval | Does retrieval add value when the document fits in 200K? |

The long-context stuffing baseline is the most important addition. If stuffing beats A-RAG on ECTSUM transcripts (which are 5K–15K words and comfortably fit in 200K context), then the business case for agentic retrieval on shorter documents is weak and the framework should pivot to long-document-first datasets like DocFinQA. If A-RAG wins despite having the same underlying LLM, it demonstrates that guided retrieval improves focus and reduces distraction from irrelevant context.

---

## Architecture Changes

### New: Task Type in Config

Add `task_type` field to `AgentConfig`:
```yaml
# configs/ectsum.yaml
task_type: summarization            # new field; defaults to "qa" for all existing configs
summarization_style: earnings_call  # earnings_call | section | query_focused | comparison
target_length: 75                   # approximate word count for the output summary
max_loops: 22                       # more loops needed vs QA
```

### New: Summarization System Prompt (`src/arag/agent/prompts.py`)

Separate from the QA system prompt. Key differences:
- Instructs the agent to retrieve **broadly** across multiple document sections before synthesizing
- Explicitly says: do not stop after the first relevant chunk — continue until all major sections are covered
- Specifies output structure: opening context sentence → key body points → outlook/closing if applicable
- Anti-hallucination: only include claims supported by retrieved chunks
- Keep `SYSTEM_PROMPT` (QA) entirely unchanged

### New: `chunk_summarize` Tool (Optional, Phase 2)

A fourth tool that generates a compressed summary of one or more chunks — prevents token budget exhaustion during broad retrieval on long transcripts.

```python
# src/arag/tools/chunk_summarize.py
class ChunkSummarizeTool:
    name = "chunk_summarize"
    # Input: chunk_ids (list), focus (optional string topic)
    # Output: {"summaries": [{"chunk_id": N, "summary": "...100-word compressed text..."}]}
    # Implementation: bounded Haiku call, temperature 0
```

### Modified: `batch_runner.py`

- Dispatch on `config.task_type` to select system prompt
- Add `cost_usd` and `latency_ms` fields to each prediction record (see Data Schema below)

### Modified: `eval.py`

- Add `--task-type` flag: routes to QA evaluator vs. summarization evaluator
- Summarization evaluator runs G-Eval, ROUGE-2, BERTScore, retrieval P/R, and tags error codes

---

## Datasets

### Primary: ECTSUM (Earnings Call Transcript Summarization)

- **Source**: [GitHub — rajdeep-biswas/ECTSum](https://github.com/rajdeep-biswas/ECTSum) | [EMNLP 2022](https://aclanthology.org/2022.emnlp-main.748/)
- **Size**: 495 earnings call transcripts (2017–2020) from S&P 500 companies
- **Format**: Full transcript + expert-written short summary (average 75 words) + **extracted key factual sentences** (gold evidence sentences used for retrieval P/R evaluation)
- **Why**: The only financial summarization benchmark with human-written reference summaries from domain experts. The `key_factual_sentences` field is unique — it enables retrieval quality evaluation independently of generation quality.
- **License**: MIT (verify on GitHub)

### Secondary: FinanceBench Derived QFS Tasks

Derive ~80 query-focused summarization tasks from the existing FinanceBench corpus (already indexed — zero new infra needed).

- For each of the 40 companies: "Summarize the key risk factors for {COMPANY} FY{YEAR}" and "What were the major operational highlights for {COMPANY} in FY{YEAR}?"
- Silver reference generated by Claude Sonnet over full relevant chunks; 10% spot-checked manually

### Tertiary: FiQA Summarization (Deprioritized)

- **Source**: FiQA opinion subset (666 items) — deprioritized; FinanceBench derived tasks cover the same QFS need at zero infrastructure cost

### Dataset Summary

| Dataset | Size | Type | Existing Index? | Retrieval P/R possible? |
|---|---|---|---|---|
| **ECTSUM** | 495 | Earnings call summary | No (new prepare script) | Yes — key factual sentences provided |
| **FinanceBench QFS** | ~80 | Query-focused section summary | Yes (reuse) | Partial — evidence field from original QA items |

---

## Evaluation Metrics

Summarization requires multi-dimensional evaluation. No single metric captures quality.

### Metric 1 & 2: G-Eval Faithfulness + Coverage (Primary)

**G-Eval** (Liu et al., 2023) significantly improves over a simple 1–5 prompt by generating evaluation steps via chain-of-thought before scoring. This makes scores more stable, reproducible, and debuggable — you can inspect the reasoning steps, not just the number.

**Protocol for each dimension:**

```
Step A — Generate evaluation criteria (one call per dimension, cached):
  Prompt: "You are designing an evaluation rubric for financial summarization.
  List 5 specific, measurable criteria for evaluating [faithfulness | coverage]
  of a financial earnings call summary. Be concrete about what score 1, 3, 5 means."
  → Outputs criteria list (generated once, reused across all predictions)

Step B — Score against criteria (one call per prediction):
  Prompt: "Using the following evaluation criteria: {criteria}

  Evaluate this summary step by step:

  Source context: {retrieved_chunks}          ← for faithfulness
  Reference summary: {reference}              ← for coverage
  Predicted summary: {predicted}

  For each criterion, reason about whether the summary satisfies it.
  Then provide a final integer score from 1–5.

  Format your response as:
  Criterion 1: [reasoning]
  Criterion 2: [reasoning]
  ...
  Final score: [1-5]"
  → Parse final score; store full reasoning in eval record for debugging
```

**Why G-Eval over simple scoring**:
- Chain-of-thought forces the judge to consider each criterion explicitly → reduces positional and length biases
- The reasoning steps are stored and inspectable — when a summary scores 2 on faithfulness, you can see exactly which claims the judge flagged as unsupported
- More reproducible: CoT anchors the score to named criteria rather than holistic impression

**Score**: Mean G-Eval score (1–5) per dimension; also report % with score ≥ 4 and distribution histogram

**Error code assignment**: In the same G-Eval call for faithfulness, if score < 3, instruct the judge to assign one or more error codes from the taxonomy (H, N, O, P, IR, IC, V) based on the reasoning it generated.

### Metric 3: Retrieval Precision/Recall (Primary for Retrieval Quality)

ECTSUM provides `key_factual_sentences` — the specific sentences from the transcript that an expert analyst selected as the essential facts. These serve as pseudo-gold evidence.

**Method**:
1. For each ECTSUM prediction, extract the set of chunk IDs that the agent actually read (`chunk_read` calls in the trace)
2. For each gold key sentence, check which chunk it falls in (pre-computed during `prepare_ectsum.py`)
3. Compute:
   - **Retrieval Recall** = fraction of gold-evidence chunks that the agent retrieved
   - **Retrieval Precision** = fraction of agent-retrieved chunks that contain at least one gold key sentence

**Why this matters**: Retrieval P/R is independent of generation quality. A system can have high faithfulness (only says what it retrieved) but low recall (missed the most important chunks). This metric diagnoses whether coverage failures are a retrieval problem or a synthesis problem.

**Output**: Added to `predictions.eval.jsonl` as `retrieval_recall` and `retrieval_precision`.

**Run on**: ECTSUM (key sentences provided). FinanceBench QFS uses the `evidence` field from original QA items as a partial proxy.

### Metric 4: ROUGE-2 (Secondary)

Standard ROUGE-2 F1 between predicted and reference summary. Enables comparison to published ECTSUM baselines. Treat as a lower bound — paraphrased-but-accurate content is penalised. Library: `rouge-score`.

### Metric 5: BERTScore F1 (Secondary)

Semantic similarity between predicted and reference using `ProsusAI/finbert` (financial-domain BERT). More robust than ROUGE for equivalent financial phrasings. Library: `bert-score`. Report P/R/F; use F1 as the headline number. Run in batch mode to amortise model load cost.

### Metric 6: Factual Consistency % (Optional, Phase 2)

Extract numerical claims from the predicted summary; verify each against source chunks via a dedicated Haiku call. % of claims verified correct. Catches the Numerical Error (N) taxonomy code programmatically.

### Metrics Summary

| Metric | Type | What it catches | Priority |
|---|---|---|---|
| **G-Eval Faithfulness** | LLM-judge w/ CoT (1–5) | Hallucination + unsupported claims | Primary |
| **G-Eval Coverage** | LLM-judge w/ CoT (1–5) | Omission of key facts | Primary |
| **Retrieval Recall/Precision** | Chunk-level overlap | Whether agent retrieved the right evidence | Primary |
| **ROUGE-2 F1** | String overlap | N-gram precision/recall vs reference | Secondary |
| **BERTScore F1** | Embedding similarity | Paraphrase-robust semantic overlap | Secondary |
| **Factual Consistency %** | LLM-judge (per claim) | Numerical accuracy | Optional Phase 2 |

---

## Data Schema

### Extended `predictions.jsonl` (per prediction record)

```json
{
  "id": "ectsum_0001",
  "source": "ectsum",
  "question": "Summarize the key highlights from {COMPANY}'s Q3 2023 earnings call.",
  "ground_truth": "{expert_reference_summary}",
  "predicted": "...",
  "loops": 14,
  "input_tokens": 84200,
  "output_tokens": 312,
  "max_loops_reached": false,
  "word_count": 78,
  "cost_usd": 0.0412,
  "latency_ms": 18340,
  "trace": [...]
}
```

**New fields vs QA schema**:
- `word_count`: actual word count of predicted summary
- `cost_usd`: total API cost for this prediction (input tokens × price + output tokens × price; uses published Anthropic pricing at run time)
- `latency_ms`: wall-clock time from first API call to final answer

### Extended `predictions.eval.jsonl` (per eval record)

```json
{
  "id": "ectsum_0001",
  "geval_faithfulness": 4,
  "geval_faithfulness_reasoning": "Criterion 1: Revenue figure $2.3B matches chunk 42. Criterion 2: ...",
  "geval_coverage": 3,
  "geval_coverage_reasoning": "Criterion 1: Guidance section not mentioned. ...",
  "retrieval_recall": 0.71,
  "retrieval_precision": 0.55,
  "rouge2_f1": 0.298,
  "bertscore_f1": 0.841,
  "word_count": 78,
  "loops": 14,
  "cost_usd": 0.0412,
  "latency_ms": 18340,
  "error_codes": []
}
```

---

## Cost + Latency Tracking

### Why it matters

The business case for A-RAG vs alternatives rests partly on cost efficiency. A-RAG uses 22× more LLM calls than naive RAG per question. Long-context stuffing uses 1 call but with a much larger input. This must be quantified concretely.

### Implementation

**In `batch_runner.py`**:
- Record `start_time = time.monotonic()` before the first API call per question
- Accumulate `input_tokens` and `output_tokens` across all loops (already tracked)
- On completion: `latency_ms = int((time.monotonic() - start_time) * 1000)`
- Compute `cost_usd` using a pricing table in `src/arag/config.py`:
  ```python
  PRICING = {
      "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},   # per million tokens
      "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
  }
  ```

**In the analysis report**:
- Cost per question: A-RAG vs Naive RAG vs Long-Context Stuffing
- Cost per accuracy point: `cost_per_question / geval_coverage_score` — the efficiency frontier
- Latency distribution: P50/P90/P99 per system (critical for interactive use cases)
- Aggregate table:

| System | Mean cost/Q | P90 latency | G-Eval Coverage | Cost per coverage point |
|---|---|---|---|---|
| A-RAG | $0.042 | 22s | 3.9 | $0.011 |
| Naive RAG | $0.003 | 4s | 2.5 | $0.001 |
| Long-Context Stuffing | $0.018 | 6s | 3.2 | $0.006 |

---

## Baselines

### Baseline 1: Naive RAG (`baselines/naive_rag_summary.py`)

Semantic search → top-k chunks → concatenate → single LLM call with summarization prompt. Already planned. Expected weakness: top-k=5 misses sections not ranked highly by the query embedding.

### Baseline 2: Long-Context Stuffing (`baselines/long_context_summary.py`)

**No retrieval at all.** Concatenate the full document (all chunks for that transcript/filing) into a single context window and call the LLM once with the summarization prompt.

```python
# baselines/long_context_summary.py
def run_stuffing(question, dataset, config):
    all_chunks = load_all_chunks(config.chunks_file, doc_id=question["doc_id"])
    full_context = "\n\n---\n\n".join(chunk["text"] for chunk in all_chunks)
    # Single API call — no tools, no retrieval
    response = anthropic_client.messages.create(
        model=config.model,
        system=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{full_context}\n\n{question['question']}"}],
        max_tokens=config.max_tokens,
    )
    return {"predicted": response.content[0].text, "loops": 1, "cost_usd": ..., "latency_ms": ...}
```

**Applicability**: Only valid for documents that fit within the model context window. For ECTSUM transcripts (5K–15K words ≈ 7K–20K tokens), this is comfortably within the 200K window. For DocFinQA full 10-K filings (123K average words ≈ 160K+ tokens), stuffing may not fit — flag and skip those cases.

**What this baseline tells you**:
- If stuffing ≈ A-RAG: retrieval adds no value for short-document summarization; A-RAG's advantage only appears on long documents where stuffing fails
- If A-RAG > stuffing: the iterative, focused retrieval improves synthesis quality even when the whole document could fit in context (probably due to reduced distraction from irrelevant sections)
- If stuffing > A-RAG: the agent's retrieval is missing important sections that are needed for a complete summary (coverage failure driven by retrieval, not generation)

**Config** (`configs/ectsum_stuffing.yaml`): Same as ECTSUM but `task_type: stuffing` and no index required (uses raw chunks only).

---

## New File Structure

```
ARagPoc/
├── src/arag/
│   ├── agent/
│   │   ├── loop.py               # MODIFIED: dispatch on task_type; add cost/latency tracking
│   │   └── prompts.py            # MODIFIED: add SUMMARIZATION_SYSTEM_PROMPT, G-Eval prompts
│   ├── tools/
│   │   └── chunk_summarize.py    # NEW: compressed chunk summarization tool (optional)
│   └── config.py                 # MODIFIED: task_type, summarization_style, target_length, PRICING table
│
├── scripts/
│   ├── prepare_ectsum.py         # NEW: download + index ECTSUM; pre-compute gold-chunk map
│   ├── derive_financebench_sum.py # NEW: derive QFS tasks from existing FinanceBench index
│   └── eval.py                   # MODIFIED: G-Eval, retrieval P/R, error taxonomy tagging
│
├── baselines/
│   ├── naive_rag_summary.py      # NEW: top-k retrieval → one-shot summarization
│   └── long_context_summary.py   # NEW: full document stuffing → one-shot summarization
│
├── configs/
│   ├── ectsum.yaml               # NEW: A-RAG on ECTSUM
│   ├── ectsum_naive.yaml         # NEW: Naive RAG on ECTSUM
│   ├── ectsum_stuffing.yaml      # NEW: Long-context stuffing on ECTSUM
│   └── financebench_sum.yaml     # NEW: A-RAG on derived QFS tasks
│
├── data/
│   ├── ectsum/
│   │   ├── chunks.json           # NEW
│   │   ├── index/                # NEW (FAISS)
│   │   ├── questions.json        # NEW
│   │   └── gold_chunk_map.json   # NEW: key_sentence → chunk_id mapping for retrieval P/R
│   └── financebench/             # EXISTING (reused)
│
├── results/
│   ├── ectsum/                   # A-RAG predictions + eval
│   ├── ectsum_naive/             # Naive RAG predictions + eval
│   ├── ectsum_stuffing/          # Stuffing predictions + eval
│   └── summarization_report.md   # Head-to-head comparison
│
└── tests/
    ├── test_summarization_eval.py # NEW: G-Eval prompts, retrieval P/R, error taxonomy
    └── test_prompts.py            # NEW: summarization prompt selection, no QA regression
```

---

## Implementation Phases

### Phase 1 — Config, Prompts & Cost/Latency Tracking (2–3 days)

**Goal**: Make the agent task-aware and instrument every prediction with cost and latency. No new datasets yet.

1. **`src/arag/config.py`**:
   - Add `task_type: str = "qa"`, `summarization_style: str = "earnings_call"`, `target_length: int = 200`
   - Add `PRICING` dict keyed by model name (input/output cost per million tokens)

2. **`src/arag/agent/prompts.py`**:
   - Add `SUMMARIZATION_SYSTEM_PROMPT` (broad retrieval, anti-hallucination, format instructions)
   - Keep existing `SYSTEM_PROMPT` (QA) unchanged

3. **`src/arag/agent/loop.py`**:
   - Select system prompt based on `config.task_type`
   - Add `start_time` tracking; compute `latency_ms` and `cost_usd` before returning result

4. **`batch_runner.py`**: Write `cost_usd`, `latency_ms`, `word_count` into prediction records

5. **Tests**: `tests/test_prompts.py` — verify prompt dispatch; confirm no QA test regression

**Deliverable**: Single summarization question runs end-to-end on existing FinanceBench index; output record includes cost and latency fields.

---

### Phase 2 — G-Eval + Error Taxonomy in `eval.py` (2–3 days)

**Goal**: Implement G-Eval for both dimensions, retrieval P/R scaffolding, and error taxonomy tagging.

1. **`scripts/eval.py`** — add `--task-type summarization` path:
   - `generate_geval_criteria(dimension)`: One-time call to generate criteria for faithfulness / coverage; cache to a JSON file so it isn't regenerated on every run
   - `evaluate_geval(prediction, reference, retrieved_chunks, criteria, dimension)`: Per-prediction CoT scoring call; parse final score and reasoning
   - `assign_error_codes(faithfulness_reasoning, coverage_reasoning)`: Extract taxonomy codes (H, N, O, P, IR, IC, V) from G-Eval reasoning text using a lightweight Haiku call — only fired when faithfulness < 3 or coverage < 3
   - `compute_retrieval_pr(trace, gold_chunk_ids)`: Precision/recall from agent trace + pre-computed gold chunk map
   - ROUGE-2 and BERTScore as before

2. **G-Eval prompt templates** in `src/arag/agent/prompts.py`:
   ```
   GEVAL_CRITERIA_PROMPT (one per dimension):
   "You are designing a rubric for evaluating [faithfulness | coverage] of a financial
   earnings call summary. List exactly 5 specific, measurable criteria. For each criterion,
   define what a score of 1, 3, and 5 looks like with a concrete example."

   GEVAL_SCORE_PROMPT:
   "Evaluation criteria: {criteria}

   Source context (retrieved chunks): {retrieved_chunks}
   Reference summary: {reference}
   Predicted summary: {predicted}

   Evaluate the predicted summary step by step against each criterion.
   Criterion 1: [your reasoning]
   Criterion 2: [your reasoning]
   Criterion 3: [your reasoning]
   Criterion 4: [your reasoning]
   Criterion 5: [your reasoning]
   Final score: [integer 1-5]"

   ERROR_CODE_PROMPT (only when score < 3):
   "Based on the following evaluation reasoning, assign one or more error codes.
   Codes: H=hallucination, N=numerical error, O=omission, P=premature termination,
   IR=irrelevant retrieval, IC=incoherence, V=verbosity.
   Reasoning: {geval_reasoning}
   Respond with a comma-separated list of codes, e.g.: H,O"
   ```

3. **Cached criteria**: Stored in `results/.geval_criteria_cache.json` — regenerate only if model or prompt changes

4. **Tests** in `tests/test_summarization_eval.py`:
   - Mock perfect summary → faithfulness=5, coverage=5, no error codes
   - Mock hallucinated number → faithfulness≤2, error code includes H
   - Mock omission of guidance section → coverage≤3, error code includes O
   - Mock trace with correct chunk IDs → retrieval_recall=1.0
   - Mock trace missing gold chunks → retrieval_recall < 1.0

**Deliverable**: `uv run python scripts/eval.py --predictions results/test/predictions.jsonl --task-type summarization` produces all metrics including G-Eval reasoning stored in eval records.

---

### Phase 3 — ECTSUM Dataset Pipeline (2–3 days)

**Goal**: `data/ectsum/chunks.json`, `data/ectsum/index/`, `data/ectsum/questions.json`, and `data/ectsum/gold_chunk_map.json` ready.

1. **`scripts/prepare_ectsum.py`**:
   - Download ECTSUM (transcripts + reference summaries + key factual sentences)
   - Chunk each transcript with metadata prefix: `[COMPANY: {ticker} | TYPE: EARNINGS_CALL | PERIOD: {Q}_{YEAR}]`
   - Build FAISS sentence-level index
   - **Build `gold_chunk_map.json`**: For each key factual sentence in ECTSUM, find which chunk it falls in (string match post-chunking) — stored as `{question_id: [chunk_id, ...]}` for retrieval P/R eval
   - Convert to `questions.json` with `evidence` = key factual sentences, `summarization_style: earnings_call`
   - `--limit N` flag for dev runs

2. **`configs/ectsum.yaml`**: `task_type: summarization`, `max_loops: 22`, `target_length: 75`

3. **Validation**: Run 5 questions; inspect traces; verify `gold_chunk_map.json` coverage (all key sentences should map to a chunk)

**Deliverable**: `prepare_ectsum.py` completes; 20-question dev run with retrieval P/R in eval output.

---

### Phase 4 — FinanceBench Derived QFS Tasks (1–2 days)

**Goal**: ~80 query-focused summarization tasks from existing FinanceBench corpus; zero new indexing.

1. **`scripts/derive_financebench_sum.py`**:
   - Generate two questions per company (risk factors + operational highlights)
   - Reference: Claude Sonnet over full relevant chunks; 10% spot-checked manually
   - Output: `data/financebench_sum/questions.json` (reuses existing `data/financebench/` index)

2. **`configs/financebench_sum.yaml`**: Points to existing index; `task_type: summarization`, `summarization_style: query_focused`

**Deliverable**: End-to-end run on derived tasks with no new downloads.

---

### Phase 5 — Three-Baseline Setup (2 days)

**Goal**: All three systems run on all datasets and produce comparable output schemas.

1. **`baselines/naive_rag_summary.py`**:
   - Semantic top-k → concatenate → single LLM call
   - Outputs: same schema as A-RAG predictions (loops=1, cost_usd, latency_ms included)

2. **`baselines/long_context_summary.py`**:
   - Load all chunks for the target document → full-context single call
   - Check token count first: if > 180K tokens, flag as `skipped: true` (too long for stuffing) and record reason
   - Outputs: same schema (loops=1, no retrieval trace)

3. Run all three on ECTSUM dev split (20 questions) to verify schema compatibility before full runs.

**Deliverable**: Three prediction files per dataset; all parseable by `eval.py`.

---

### Phase 6 — Full Evaluation Run & Analysis (2–3 days)

**Goal**: Head-to-head comparison report across all three systems on ECTSUM and FinanceBench QFS.

1. **Full batch inference** (all three systems on both datasets)
2. **Full eval** with G-Eval, retrieval P/R, ROUGE-2, BERTScore, error taxonomy
3. **Analysis report** (`results/summarization_report.md`):
   - Headline metrics table (A-RAG vs Naive RAG vs Long-Context Stuffing)
   - Cost + latency table with cost-per-coverage-point efficiency metric
   - Error taxonomy breakdown: bar chart of H/N/O/P/IR/IC/V frequency per system
   - Retrieval P/R analysis: loop count vs retrieval recall correlation
   - Qualitative examples: 2 wins for A-RAG, 1 case where stuffing wins, 1 failure with G-Eval reasoning shown
   - Key finding statement: "Retrieval adds/does not add value for [short/long] documents because..."

**Expected results hypothesis**:

| Metric | A-RAG | Naive RAG | Long-Context Stuffing |
|---|---|---|---|
| G-Eval Faithfulness (1–5) | 4.2 | 3.8 | 4.3 (all context available) |
| G-Eval Coverage (1–5) | 3.9 | 2.4 | 3.5 (distracted by irrelevant) |
| Retrieval Recall | 0.74 | 0.41 | N/A |
| ROUGE-2 F1 | 0.31 | 0.19 | 0.27 |
| Mean cost/Q | $0.042 | $0.003 | $0.018 |
| P90 latency | 22s | 4s | 6s |

---

### Phase 7 — Streamlit UI Extensions (1–2 days)

**Goal**: Summarization tasks first-class in the demo UI.

1. **`app/streamlit_app.py`** updates:
   - Dataset selector includes ECTSUM and FinanceBench QFS
   - Metrics panel: G-Eval faithfulness/coverage scores, retrieval recall, cost/latency
   - Side-by-side tab: A-RAG vs Naive RAG vs Long-Context Stuffing summary (3-column layout)
   - Trace view: section coverage indicator — which document sections (Q&A, guidance, financials) were touched
   - G-Eval reasoning expandable panel (click to see judge's step-by-step rationale)

---

## Key Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| G-Eval criteria generation is non-deterministic | Generate criteria once at run start; cache in `results/.geval_criteria_cache.json`; regenerate only on explicit `--regenerate-criteria` flag |
| G-Eval CoT scoring is ~3× more expensive than simple scoring | Use Haiku for scoring (not Sonnet); CoT reasoning is ~200 tokens extra per call; tolerable at scale |
| Long-context stuffing may not fit some ECTSUM transcripts | Check token count pre-call; flag oversized inputs as `skipped`; report what % of the dataset is stuffable |
| ECTSUM key sentences don't always align cleanly to chunk boundaries | Use substring search with ±50 char tolerance in `gold_chunk_map.json` builder; log alignment failures |
| Agent stops too early (QA habit) | Summarization system prompt explicitly enforces broad coverage; `max_loops: 22`; monitor premature termination (P) error code rate |
| G-Eval reasoning parsing is brittle | Use strict format enforcement in the scoring prompt ("Final score: [integer]"); add regex fallback + log when parsing fails |
| Silver reference quality for FinanceBench derived tasks | Generate with Sonnet; spot-check 10% manually before treating as ground truth |
| BERTScore compute at 495-item ECTSUM scale | Run in batch mode; cache FinBERT embeddings; ~10 min on CPU, ~90s on GPU |
| Error code assignment adds judge API calls | Only fire for low-scoring predictions (faithfulness < 3 or coverage < 3); expected ~20–30% of predictions |

---

## Dependencies to Add

```toml
# pyproject.toml additions
rouge-score = ">=0.1.2"
bert-score = ">=0.3.13"
```

Both are CPU-compatible. FinBERT model (~440MB) downloads automatically on first `bert_score` call.

---

## Success Criteria

| Criterion | Target |
|---|---|
| All existing QA tests still pass | 100% pass |
| G-Eval produces parseable scores on 100% of predictions | No unparseable outputs |
| Retrieval P/R computed for all ECTSUM predictions | Recall and precision in every eval record |
| `cost_usd` and `latency_ms` present in all prediction records | 100% coverage |
| Error taxonomy covers ≥ 95% of low-scoring predictions (score < 3) | ≥ 95% have ≥ 1 error code |
| Long-context stuffing pipeline runs on ECTSUM without errors | No crashes; skipped records flagged cleanly |
| A-RAG retrieval recall > Naive RAG implied recall by ≥ 0.25 | Validates iterative retrieval advantage |
| G-Eval coverage: A-RAG > Naive RAG by ≥ 0.8 points | Validates multi-section synthesis advantage |

---

## Milestones

| Milestone | Deliverable | Phase |
|---|---|---|
| **S1** — Infra ready | `task_type`, cost/latency tracking, summarization prompt; QA tests pass | 1 |
| **S2** — Metrics working | G-Eval + error taxonomy + retrieval P/R scaffold validated on toy examples | 2 |
| **S3** — ECTSUM pipeline | `prepare_ectsum.py` + `gold_chunk_map.json` complete; 20-Q dev run succeeds | 3 |
| **S4** — Derived tasks ready | FinanceBench QFS questions + silver references generated | 4 |
| **S5** — All three baselines running | Naive RAG + long-context stuffing predictions on all datasets | 5 |
| **S6** — Full eval report | Head-to-head table across 3 systems: metrics, cost, error taxonomy | 6 |
| **S7** — UI updated | Streamlit shows 3-system comparison + G-Eval reasoning panel | 7 |

---

## Out of Scope (for This Phase)

- **Abstractive vs extractive comparison**: A-RAG always produces abstractive summaries; extractive baselines not planned
- **Multi-document summarization**: Cross-filing comparison requiring multiple indices
- **Streaming summaries**: Progressive output during generation
- **Human evaluation**: LLM-judge is the proxy for this POC
- **FiQA dataset preparation**: Derived FinanceBench tasks cover the QFS use case at zero infra cost

---

## References

- ECTSUM: [arXiv:2210.12569](https://arxiv.org/abs/2210.12569) | [GitHub — rajdeep-biswas/ECTSum](https://github.com/rajdeep-biswas/ECTSum)
- G-Eval: [arXiv:2303.16634](https://arxiv.org/abs/2303.16634) — Liu et al., 2023
- ROUGE: [lin-2004-rouge](https://aclanthology.org/W04-1013/)
- BERTScore: [arXiv:1904.09675](https://arxiv.org/abs/1904.09675)
- FinBERT: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- SummaC (faithfulness evaluation): [arXiv:2111.09525](https://arxiv.org/abs/2111.09525) — approach adapted for LLM-judge
