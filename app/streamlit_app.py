"""
A-RAG Financial Demo — Streamlit UI

Run with:
    uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.agent.loop import AgentLoop
from src.arag.config import AgentConfig, get_api_key
from src.arag.tools.chunk_read import ChunkReadTool
from src.arag.tools.keyword_search import KeywordSearchTool
from src.arag.tools.semantic_search import SemanticSearchTool
from baselines.naive_rag import run_naive_rag
from baselines.naive_rag_summary import run_naive_rag_summary
from baselines.long_context_summary import run_stuffing, _load_chunks as _load_all_chunks

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="A-RAG Financial Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = {
    "FinanceBench": "financebench",
    "FinDER": "finder",
    "FinQA": "finqa",
    "ECTSUM": "ectsum",
    "FinanceBench QFS": "financebench_sum",
}

_EXAMPLE_FALLBACKS = {
    "financebench": [
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
        "Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023?",
        "What drove operating margin change as of FY2022 for 3M?",
        "What is Amazon's FY2017 days payable outstanding (DPO)?",
        "Does Adobe have an improving operating margin profile as of FY2022?",
    ],
    "finder": [
        "What is the company's current ratio?",
        "What were the key risk factors disclosed in the most recent 10-K?",
    ],
    "finqa": [
        "What was the percentage change in operating expenses from 2021 to 2022?",
        "How did the company's net income change between the two most recent fiscal years?",
    ],
    "ectsum": [
        "Summarize the key highlights from this earnings call.",
    ],
    "financebench_sum": [
        "Summarize the key financial highlights.",
    ],
}

TOOL_ICONS = {
    "keyword_search": "🔑",
    "semantic_search": "🔍",
    "chunk_read": "📄",
}

TOOL_COLOURS = {
    "keyword_search": "#f0f4ff",
    "semantic_search": "#f0fff4",
    "chunk_read": "#fff8f0",
}

# Error code descriptions for the taxonomy table
_ERROR_CODES = [
    ("H",  "Hallucination"),
    ("N",  "Numerical Error"),
    ("O",  "Omission"),
    ("P",  "Premature Term."),
    ("IR", "Irrel. Retrieval"),
    ("V",  "Verbosity"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_ready(dataset: str) -> bool:
    try:
        config = AgentConfig.from_yaml(f"configs/{dataset}.yaml")
        return (
            Path(config.chunks_file).exists()
            and Path(config.index_dir, "sentences.faiss").exists()
        )
    except Exception:
        return False


def _is_summarization(dataset: str) -> bool:
    """Return True if this dataset uses task_type: summarization."""
    try:
        config = AgentConfig.from_yaml(f"configs/{dataset}.yaml")
        return getattr(config, "task_type", "qa") == "summarization"
    except Exception:
        return False


def _load_example_questions(dataset: str) -> list[str]:
    try:
        path = f"data/{dataset}/questions.json"
        qs = json.load(open(path))
        seen: list[str] = []
        for q in qs:
            text = q.get("question", "").strip()
            if text and len(text) < 200:
                seen.append(text)
            if len(seen) >= 5:
                break
        return seen
    except Exception:
        return _EXAMPLE_FALLBACKS.get(dataset, [])


def _load_eval_results(dataset: str) -> tuple[list[dict], list[dict]] | None:
    """Return (predictions, eval_rows) or None if not available."""
    pred_path = Path(f"results/{dataset}/predictions.jsonl")
    eval_path = Path(f"results/{dataset}/predictions.eval.jsonl")
    if not pred_path.exists() or not eval_path.exists():
        return None
    preds = [json.loads(l) for l in pred_path.open() if l.strip()]
    evals = [json.loads(l) for l in eval_path.open() if l.strip()]
    return preds, evals


def _load_naive_eval_results(dataset: str) -> tuple[list[dict], list[dict]] | None:
    pred_path = Path(f"results/{dataset}_naive/predictions.jsonl")
    eval_path = Path(f"results/{dataset}_naive/predictions.eval.jsonl")
    if not pred_path.exists() or not eval_path.exists():
        return None
    preds = [json.loads(l) for l in pred_path.open() if l.strip()]
    evals = [json.loads(l) for l in eval_path.open() if l.strip()]
    return preds, evals


def _load_stuffing_eval_results(dataset: str) -> tuple[list[dict], list[dict]] | None:
    pred_path = Path(f"results/{dataset}_stuffing/predictions.jsonl")
    eval_path = Path(f"results/{dataset}_stuffing/predictions.eval.jsonl")
    if not pred_path.exists() or not eval_path.exists():
        return None
    preds = [json.loads(l) for l in pred_path.open() if l.strip()]
    evals = [json.loads(l) for l in eval_path.open() if l.strip()]
    return preds, evals


def _mean(records: list[dict], key: str) -> float:
    vals = [r.get(key) for r in records if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _pct_ge4(records: list[dict], key: str) -> float:
    vals = [r.get(key) for r in records if r.get(key) is not None]
    return sum(1 for v in vals if v >= 4) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading agent and index…")
def load_agent(dataset: str) -> tuple[AgentLoop, AgentConfig]:
    config = AgentConfig.from_yaml(f"configs/{dataset}.yaml")
    api_key = get_api_key()
    agent = AgentLoop(
        config=config,
        keyword_tool=KeywordSearchTool(chunks_file=config.chunks_file),
        semantic_tool=SemanticSearchTool(
            chunks_file=config.chunks_file,
            index_dir=config.index_dir,
            embedding_model=config.embedding_model,
            device=config.embedding_device,
        ),
        chunk_tool=ChunkReadTool(chunks_file=config.chunks_file),
        api_key=api_key,
    )
    return agent, config


@st.cache_resource(show_spinner="Loading summarization resources…")
def load_sum_resources(dataset: str):
    """
    Load all resources needed for the live summarization demo:
    A-RAG agent, Naive RAG tools, Stuffing chunks, and Anthropic client.
    """
    config = AgentConfig.from_yaml(f"configs/{dataset}.yaml")
    api_key = get_api_key()
    client = anthropic.Anthropic(api_key=api_key)

    # Shared semantic search tool (thread-safe for read-only inference)
    sem_tool = SemanticSearchTool(
        chunks_file=config.chunks_file,
        index_dir=config.index_dir,
        embedding_model=config.embedding_model,
        device=config.embedding_device,
    )
    # Dedicated chunk tool for naive RAG (independent C_read state)
    chunk_tool_naive = ChunkReadTool(chunks_file=config.chunks_file)

    # All chunks pre-loaded for stuffing baseline
    all_chunks = _load_all_chunks(config.chunks_file)

    # A-RAG agent with its own tool instances
    agent = AgentLoop(
        config=config,
        keyword_tool=KeywordSearchTool(chunks_file=config.chunks_file),
        semantic_tool=SemanticSearchTool(
            chunks_file=config.chunks_file,
            index_dir=config.index_dir,
            embedding_model=config.embedding_model,
            device=config.embedding_device,
        ),
        chunk_tool=ChunkReadTool(chunks_file=config.chunks_file),
        api_key=api_key,
    )

    return agent, config, sem_tool, chunk_tool_naive, all_chunks, client


# ---------------------------------------------------------------------------
# Sub-renderers — trace (shared by QA and summarization)
# ---------------------------------------------------------------------------

def render_trace(trace: list[dict]) -> None:
    """Render the agent's step-by-step retrieval trace."""
    if not trace:
        return

    tool_counts: dict[str, int] = {}
    for step in trace:
        tool_counts[step["tool"]] = tool_counts.get(step["tool"], 0) + 1

    cols = st.columns(len(tool_counts) + 1)
    cols[0].markdown("**Tool calls:**")
    for i, (tool, count) in enumerate(tool_counts.items(), 1):
        cols[i].markdown(f"{TOOL_ICONS.get(tool, '⚙️')} **{tool}** × {count}")

    st.markdown("---")

    for step in trace:
        icon = TOOL_ICONS.get(step["tool"], "⚙️")
        label = f"Loop {step['loop']} — {icon} `{step['tool']}`"
        with st.expander(label, expanded=False):
            left, right = st.columns(2)
            with left:
                st.markdown("**Input**")
                inp = step["input"]
                if step["tool"] == "keyword_search":
                    st.markdown(f"Keywords: `{inp.get('keywords', [])}`")
                    st.markdown(f"top_k: `{inp.get('top_k', '')}`")
                elif step["tool"] == "semantic_search":
                    st.markdown(f"Query: *{inp.get('query', '')}*")
                    st.markdown(f"top_k: `{inp.get('top_k', '')}`")
                else:
                    st.markdown(f"Chunk IDs: `{inp.get('chunk_ids', [])}`")

            with right:
                st.markdown("**Results**")
                output = step["output"]
                if "results" in output:
                    for r in output["results"][:3]:
                        text = r.get("text", "")
                        if len(text) > 300:
                            text = text[:300] + "…"
                        score = r.get("score", r.get("keyword_score", ""))
                        score_str = f" · score={score:.3f}" if isinstance(score, float) else ""
                        st.markdown(
                            f"**Chunk {r.get('chunk_id', '')}**{score_str}  \n"
                            f"<small>{text}</small>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("---")
                elif "error" in output:
                    st.error(output["error"])
                else:
                    st.json(output)


def render_section_coverage(trace: list[dict]) -> None:
    """
    Show a section coverage indicator from the A-RAG trace.
    Extracts unique chunk IDs read and any SECTION / FILING / PERIOD metadata.
    """
    chunk_ids: list[int] = []
    sections: list[str] = []

    for step in trace:
        if step["tool"] != "chunk_read":
            continue
        for r in step["output"].get("results", []):
            cid = r.get("chunk_id")
            if cid is not None and cid not in chunk_ids:
                chunk_ids.append(cid)
            text = r.get("text", "")
            # Metadata prefix looks like: [COMPANY: X | PERIOD: Y | CHUNK: Z]
            # Capture SECTION, FILING, or PERIOD values
            for field in ("SECTION", "FILING", "PERIOD"):
                m = re.search(rf'\[.*?{field}:\s*([^\]|]+)', text)
                if m:
                    val = m.group(1).strip()[:50]
                    if val and val not in sections:
                        sections.append(val)
                    break

    if not chunk_ids:
        return

    st.markdown(
        f"**Chunks read:** `{sorted(chunk_ids)}`"
        if len(chunk_ids) <= 12
        else f"**Chunks read ({len(chunk_ids)}):** `{sorted(chunk_ids)[:12]}` …"
    )
    if sections:
        st.markdown("**Sections / periods accessed:** " + " · ".join(f"`{s}`" for s in sections[:8]))


# ---------------------------------------------------------------------------
# Sub-renderers — summarization results tab
# ---------------------------------------------------------------------------

def render_summarization_results_tab(dataset: str) -> None:
    """Render benchmark results for a summarization dataset (ECTSUM / FinanceBench QFS)."""
    arag_data     = _load_eval_results(dataset)
    naive_data    = _load_naive_eval_results(dataset)
    stuffing_data = _load_stuffing_eval_results(dataset)

    if arag_data is None:
        st.info(
            f"No evaluation results found for **{dataset}**.\n\n"
            f"Run the full evaluation pipeline:\n"
            f"```\n./scripts/run_phase6.sh --skip-prep --dev\n```"
        )
        return

    preds_a, evals_a = arag_data

    # Collect all available systems
    systems: dict[str, tuple[list[dict], list[dict]]] = {"A-RAG": (preds_a, evals_a)}
    if naive_data:
        systems["Naive RAG"] = naive_data
    if stuffing_data:
        systems["Stuffing"] = stuffing_data

    st.subheader("Summarization Benchmark Results")

    # ---- Headline metric cards ----
    metric_cols = st.columns(len(systems))
    for col, (sys_name, (preds, evals)) in zip(metric_cols, systems.items()):
        n = len(evals)
        avg_cov   = _mean(evals, "geval_coverage")
        avg_faith = _mean(evals, "geval_faithfulness")
        avg_cost  = _mean(preds, "cost_usd")
        avg_loops = _mean(preds, "loops")
        with col:
            st.markdown(f"#### {sys_name}")
            st.metric("G-Eval Coverage",     f"{avg_cov:.2f} / 5")
            st.metric("G-Eval Faithfulness", f"{avg_faith:.2f} / 5")
            st.metric("Avg cost / item",     f"${avg_cost:.4f}")
            st.metric("Avg loops",           f"{avg_loops:.1f}")
            st.caption(f"n = {n} items")

    # ---- Full metrics table ----
    st.markdown("---")
    st.markdown("#### Full Metrics Table")
    table_rows = []
    for sys_name, (preds, evals) in systems.items():
        n          = len(evals)
        skipped    = sum(1 for p in preds if p.get("skipped"))
        avg_cov    = _mean(evals, "geval_coverage")
        avg_faith  = _mean(evals, "geval_faithfulness")
        pct_cov4   = _pct_ge4(evals, "geval_coverage")
        pct_faith4 = _pct_ge4(evals, "geval_faithfulness")
        avg_rouge  = _mean(evals, "rouge2_f1")
        avg_cost   = _mean(preds, "cost_usd")
        p50_lat    = sorted(p.get("latency_ms", 0) for p in preds)[n // 2] / 1000
        avg_loops  = _mean(preds, "loops")
        avg_words  = _mean(preds, "word_count")
        table_rows.append({
            "System":              sys_name,
            "Coverage (mean)":     f"{avg_cov:.2f}",
            "Coverage (≥4)":       f"{pct_cov4:.0%}",
            "Faithfulness (mean)": f"{avg_faith:.2f}",
            "Faithfulness (≥4)":   f"{pct_faith4:.0%}",
            "ROUGE-2":             f"{avg_rouge:.3f}",
            "Avg cost":            f"${avg_cost:.4f}",
            "P50 latency":         f"{p50_lat:.1f}s",
            "Avg loops":           f"{avg_loops:.1f}",
            "Avg words":           f"{avg_words:.0f}",
            "Skipped":             skipped,
            "n":                   n,
        })
    st.dataframe(
        pd.DataFrame(table_rows).set_index("System"),
        use_container_width=True,
    )

    # ---- Error taxonomy ----
    st.markdown("---")
    st.markdown("#### Error Taxonomy")
    st.caption("Codes assigned to items scoring < 3 on faithfulness **or** coverage.")
    err_rows = []
    for sys_name, (preds, evals) in systems.items():
        counts: Counter = Counter()
        low = 0
        for e in evals:
            faith = e.get("geval_faithfulness", 5) or 5
            cov   = e.get("geval_coverage",     5) or 5
            if faith < 3 or cov < 3:
                low += 1
                for code in e.get("error_codes", []):
                    counts[code] += 1
        row: dict = {"System": sys_name, "Low-scoring": low}
        for code, label in _ERROR_CODES:
            row[f"{code} — {label}"] = counts.get(code, 0)
        err_rows.append(row)
    st.dataframe(
        pd.DataFrame(err_rows).set_index("System"),
        use_container_width=True,
    )

    # ---- A-RAG loop distribution ----
    st.markdown("---")
    st.markdown("#### Loop Distribution (A-RAG)")
    loop_dist = Counter(r.get("loops", 0) for r in preds_a)
    st.bar_chart({str(k): v for k, v in sorted(loop_dist.items())})

    # ---- Per-item browser ----
    st.markdown("---")
    st.markdown("#### Per-Item Results")

    fc, sc, sysc = st.columns([2, 1, 1])
    filter_val = fc.selectbox(
        "Filter",
        ["All", "Coverage ≥ 4", "Coverage < 3", "Faithfulness ≥ 4", "Faithfulness < 3", "Skipped"],
        key=f"sum_filter_{dataset}",
    )
    sort_val = sc.selectbox(
        "Sort by",
        ["Coverage (desc)", "Faithfulness (desc)", "Loops (desc)", "Cost (desc)", "ID"],
        key=f"sum_sort_{dataset}",
    )
    sys_val = sysc.selectbox(
        "System",
        list(systems.keys()),
        key=f"sum_sys_{dataset}",
    )

    preds_view, evals_view = systems[sys_val]
    pred_by_id = {r["id"]: r for r in preds_view}

    browser_rows: list[dict] = []
    for e in evals_view:
        p = pred_by_id.get(e["id"], {})
        browser_rows.append({
            "id":            e["id"],
            "question":      p.get("question", ""),
            "ground_truth":  p.get("ground_truth", ""),
            "predicted":     p.get("predicted", ""),
            "cov":           e.get("geval_coverage"),
            "faith":         e.get("geval_faithfulness"),
            "cov_reason":    e.get("geval_coverage_reasoning", ""),
            "faith_reason":  e.get("geval_faithfulness_reasoning", ""),
            "rouge2":        e.get("rouge2_f1", 0.0),
            "loops":         p.get("loops", 0),
            "cost":          p.get("cost_usd", 0.0),
            "words":         p.get("word_count", 0),
            "error_codes":   e.get("error_codes", []),
            "skipped":       p.get("skipped", False),
        })

    # Filter
    if filter_val == "Coverage ≥ 4":
        browser_rows = [r for r in browser_rows if (r["cov"] or 0) >= 4]
    elif filter_val == "Coverage < 3":
        browser_rows = [r for r in browser_rows if (r["cov"] or 5) < 3]
    elif filter_val == "Faithfulness ≥ 4":
        browser_rows = [r for r in browser_rows if (r["faith"] or 0) >= 4]
    elif filter_val == "Faithfulness < 3":
        browser_rows = [r for r in browser_rows if (r["faith"] or 5) < 3]
    elif filter_val == "Skipped":
        browser_rows = [r for r in browser_rows if r["skipped"]]

    # Sort
    if sort_val == "Coverage (desc)":
        browser_rows = sorted(browser_rows, key=lambda x: -(x["cov"] or 0))
    elif sort_val == "Faithfulness (desc)":
        browser_rows = sorted(browser_rows, key=lambda x: -(x["faith"] or 0))
    elif sort_val == "Loops (desc)":
        browser_rows = sorted(browser_rows, key=lambda x: -x["loops"])
    elif sort_val == "Cost (desc)":
        browser_rows = sorted(browser_rows, key=lambda x: -x["cost"])

    st.caption(f"Showing {len(browser_rows)} items")

    for row in browser_rows[:50]:
        cov   = row["cov"]
        faith = row["faith"]
        cov_icon   = "✅" if (cov   or 0) >= 4 else ("⚠️" if (cov   or 0) >= 3 else "❌")
        faith_icon = "✅" if (faith or 0) >= 4 else ("⚠️" if (faith or 0) >= 3 else "❌")
        skip_icon  = " ⏭️" if row["skipped"] else ""
        codes_str  = f" [{','.join(row['error_codes'])}]" if row["error_codes"] else ""

        with st.expander(
            f"{cov_icon}{faith_icon}{skip_icon} `{row['id']}` — "
            f"Cov={cov} Faith={faith} loops={row['loops']}{codes_str}",
            expanded=False,
        ):
            st.markdown(f"**Question:** {row['question']}")

            t_sum, t_ref, t_eval = st.tabs(["Summary", "Reference", "G-Eval Reasoning"])

            with t_sum:
                if row["skipped"]:
                    st.warning("Skipped — document too large for the stuffing baseline.")
                else:
                    preview = row["predicted"]
                    if len(preview) > 2000:
                        preview = preview[:2000] + "\n\n*(truncated — full text in predictions.jsonl)*"
                    st.markdown(preview)
                    st.caption(
                        f"Words: {row['words']} · "
                        f"Cost: ${row['cost']:.4f} · "
                        f"ROUGE-2: {row['rouge2']:.3f}"
                    )

            with t_ref:
                ref = row["ground_truth"]
                if len(ref) > 1500:
                    ref = ref[:1500] + "\n\n*(truncated)*"
                st.markdown(ref)

            with t_eval:
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown(f"**G-Eval Coverage: {cov}/5**")
                    if row["cov_reason"]:
                        st.markdown(row["cov_reason"][:2000])
                    else:
                        st.caption("No reasoning recorded.")
                with ec2:
                    st.markdown(f"**G-Eval Faithfulness: {faith}/5**")
                    if row["faith_reason"]:
                        st.markdown(row["faith_reason"][:2000])
                    else:
                        st.caption("No reasoning recorded.")


# ---------------------------------------------------------------------------
# Sub-renderers — QA results tab (unchanged)
# ---------------------------------------------------------------------------

def render_results_tab(dataset: str) -> None:
    """Render the benchmark results tab for QA datasets."""
    arag_data = _load_eval_results(dataset)
    naive_data = _load_naive_eval_results(dataset)

    if arag_data is None:
        st.info(
            f"No evaluation results found for **{dataset}**. "
            f"Run the batch evaluation first:\n\n"
            f"```\nuv run python scripts/batch_runner.py --config configs/{dataset}.yaml "
            f"--output results/{dataset} --workers 3\n"
            f"uv run python scripts/eval.py --predictions results/{dataset}/predictions.jsonl\n```"
        )
        return

    preds, evals = arag_data
    eval_by_id = {r["id"]: r for r in evals}

    n = len(preds)
    arag_correct = sum(1 for r in evals if r.get("llm_correct"))
    arag_cm = sum(1 for r in evals if r.get("contain_match"))
    avg_loops = sum(r.get("loops", 0) for r in preds) / n
    max_loops_hit = sum(1 for r in preds if r.get("max_loops_reached"))
    avg_tokens = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in preds) / n

    st.subheader("Benchmark Results")

    if naive_data:
        npreds, nevals = naive_data
        naive_correct = sum(1 for r in nevals if r.get("llm_correct"))
        naive_cm = sum(1 for r in nevals if r.get("contain_match"))
        naive_avg_tokens = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in npreds) / len(npreds)

        st.markdown("#### A-RAG vs Naive RAG")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "LLM-Accuracy",
            f"{arag_correct/n:.1%}",
            delta=f"+{(arag_correct - naive_correct)/n:.1%} vs Naive",
        )
        col2.metric(
            "Contain-Match",
            f"{arag_cm/n:.1%}",
            delta=f"+{(arag_cm - naive_cm)/n:.1%} vs Naive",
        )
        col3.metric(
            "Avg Tokens / Q",
            f"{avg_tokens:,.0f}",
            delta=f"{avg_tokens/naive_avg_tokens:.0f}× Naive",
            delta_color="inverse",
        )

        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**A-RAG**")
            st.markdown(f"- LLM-Accuracy: **{arag_correct/n:.1%}** ({arag_correct}/{n})")
            st.markdown(f"- Contain-Match: {arag_cm/n:.1%} ({arag_cm}/{n})")
            st.markdown(f"- Avg loops: {avg_loops:.1f}")
            st.markdown(f"- Max-loops hit: {max_loops_hit} ({max_loops_hit/n:.1%})")
            st.markdown(f"- Avg tokens: {avg_tokens:,.0f}")
        with colB:
            st.markdown("**Naive RAG (single-shot)**")
            st.markdown(f"- LLM-Accuracy: **{naive_correct/len(npreds):.1%}** ({naive_correct}/{len(npreds)})")
            st.markdown(f"- Contain-Match: {naive_cm/len(npreds):.1%} ({naive_cm}/{len(npreds)})")
            st.markdown(f"- Avg loops: 1")
            st.markdown(f"- Avg tokens: {naive_avg_tokens:,.0f}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LLM-Accuracy", f"{arag_correct/n:.1%}")
        col2.metric("Contain-Match", f"{arag_cm/n:.1%}")
        col3.metric("Avg Loops", f"{avg_loops:.1f}")
        col4.metric("Max-Loops Hit", f"{max_loops_hit/n:.1%}")

    # --- Loop distribution ---
    st.markdown("---")
    st.markdown("#### Loop Distribution")
    loop_dist = Counter(r.get("loops", 0) for r in preds)
    loop_data = {str(k): v for k, v in sorted(loop_dist.items())}
    st.bar_chart(loop_data)

    # --- Outcome breakdown ---
    if naive_data:
        npreds, nevals = naive_data
        naive_eval_by_id = {r["id"]: r for r in nevals}
        arag_only = sum(
            1 for r in evals
            if r.get("llm_correct") and not naive_eval_by_id.get(r["id"], {}).get("llm_correct")
        )
        both_correct = sum(
            1 for r in evals
            if r.get("llm_correct") and naive_eval_by_id.get(r["id"], {}).get("llm_correct")
        )
        both_wrong = sum(
            1 for r in evals
            if not r.get("llm_correct") and not naive_eval_by_id.get(r["id"], {}).get("llm_correct")
        )
        naive_only = sum(
            1 for r in evals
            if not r.get("llm_correct") and naive_eval_by_id.get(r["id"], {}).get("llm_correct")
        )
        st.markdown("#### Outcome Breakdown")
        outcome_cols = st.columns(4)
        outcome_cols[0].metric("A-RAG only ✅", arag_only)
        outcome_cols[1].metric("Both correct ✅", both_correct)
        outcome_cols[2].metric("Both wrong ❌", both_wrong)
        outcome_cols[3].metric("Naive only ✅", naive_only)

    # --- Per-question browser ---
    st.markdown("---")
    st.markdown("#### Per-Question Results")
    filter_col, sort_col = st.columns([2, 1])
    filter_val = filter_col.selectbox(
        "Filter",
        ["All", "A-RAG Correct", "A-RAG Wrong", "Max-loops reached"],
    )
    sort_val = sort_col.selectbox("Sort by", ["Loops (desc)", "Tokens (desc)", "ID"])

    pred_by_id = {r["id"]: r for r in preds}
    rows = []
    for e in evals:
        p = pred_by_id.get(e["id"], {})
        rows.append({
            "id": e["id"],
            "question": p.get("question", ""),
            "ground_truth": p.get("ground_truth", ""),
            "predicted": p.get("predicted", ""),
            "llm_correct": e.get("llm_correct"),
            "contain_match": e.get("contain_match"),
            "loops": p.get("loops", 0),
            "tokens": p.get("input_tokens", 0) + p.get("output_tokens", 0),
            "max_loops": p.get("max_loops_reached", False),
        })

    if filter_val == "A-RAG Correct":
        rows = [r for r in rows if r["llm_correct"]]
    elif filter_val == "A-RAG Wrong":
        rows = [r for r in rows if not r["llm_correct"]]
    elif filter_val == "Max-loops reached":
        rows = [r for r in rows if r["max_loops"]]

    if sort_val == "Loops (desc)":
        rows = sorted(rows, key=lambda x: -x["loops"])
    elif sort_val == "Tokens (desc)":
        rows = sorted(rows, key=lambda x: -x["tokens"])

    st.caption(f"Showing {len(rows)} questions")
    for row in rows[:50]:
        icon = "✅" if row["llm_correct"] else "❌"
        max_icon = " ⏱️" if row["max_loops"] else ""
        with st.expander(
            f"{icon}{max_icon} `{row['id']}` — loops={row['loops']} tokens={row['tokens']:,}",
            expanded=False,
        ):
            st.markdown(f"**Question:** {row['question']}")
            st.markdown(f"**Ground truth:** {row['ground_truth'][:300]}")
            st.markdown(f"**Predicted:** {row['predicted'][:400]}")
            st.markdown(
                f"LLM-correct: `{row['llm_correct']}` | "
                f"Contain-match: `{row['contain_match']}` | "
                f"Loops: `{row['loops']}` | "
                f"Tokens: `{row['tokens']:,}`"
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 A-RAG Demo")
    st.caption("[arXiv:2602.03442](https://arxiv.org/abs/2602.03442)")
    st.divider()

    # Dataset selector — only show datasets with data ready
    available = {k: v for k, v in DATASETS.items() if _dataset_ready(v)}
    if not available:
        st.error("No datasets ready. Run a prepare_*.py script first.")
        st.stop()

    dataset_label = st.selectbox("Dataset", list(available.keys()))
    dataset = available[dataset_label]
    is_sum = _is_summarization(dataset)

    if is_sum:
        st.caption("Summarization dataset — G-Eval scoring")
    else:
        st.caption("QA dataset — LLM-Accuracy scoring")

    st.divider()
    show_baseline = st.toggle(
        "Compare with baselines",
        value=True,
        help="Show Naive RAG and (for summarization) Long-Context Stuffing alongside A-RAG.",
    )

    st.divider()
    st.markdown("**How A-RAG works**")
    st.markdown(
        "The agent iteratively calls three retrieval tools:\n\n"
        "- 🔑 **keyword_search** — exact match scoring\n"
        "- 🔍 **semantic_search** — FAISS cosine similarity\n"
        "- 📄 **chunk_read** — full text of specific chunks\n\n"
        "It loops until it has enough evidence to answer or hits the loop limit."
    )

    # Quick eval stats in sidebar
    arag_data = _load_eval_results(dataset)
    if arag_data:
        preds_sb, evals_sb = arag_data
        n_sb = len(evals_sb)
        st.divider()
        st.markdown("**Benchmark (this dataset)**")
        if is_sum:
            avg_cov_sb   = _mean(evals_sb, "geval_coverage")
            avg_faith_sb = _mean(evals_sb, "geval_faithfulness")
            st.metric("A-RAG G-Eval Coverage",     f"{avg_cov_sb:.2f}/5 ({n_sb} items)")
            st.metric("A-RAG G-Eval Faithfulness", f"{avg_faith_sb:.2f}/5")
            naive_sum_data = _load_naive_eval_results(dataset)
            if naive_sum_data:
                _, ne_sb = naive_sum_data
                st.metric("Naive RAG Coverage", f"{_mean(ne_sb, 'geval_coverage'):.2f}/5")
            stuffing_sum_data = _load_stuffing_eval_results(dataset)
            if stuffing_sum_data:
                _, se_sb = stuffing_sum_data
                st.metric("Stuffing Coverage", f"{_mean(se_sb, 'geval_coverage'):.2f}/5")
        else:
            correct_sb = sum(1 for r in evals_sb if r.get("llm_correct"))
            st.metric("A-RAG LLM-Accuracy", f"{correct_sb/n_sb:.1%} ({correct_sb}/{n_sb})")
            naive_data_sb = _load_naive_eval_results(dataset)
            if naive_data_sb:
                npreds_sb, nevals_sb = naive_data_sb
                nc_sb = sum(1 for r in nevals_sb if r.get("llm_correct"))
                st.metric("Naive RAG LLM-Accuracy", f"{nc_sb/len(npreds_sb):.1%} ({nc_sb}/{len(npreds_sb)})")


# ---------------------------------------------------------------------------
# Main content — tabs
# ---------------------------------------------------------------------------

tab_demo, tab_results = st.tabs(["💬 Demo", "📊 Benchmark Results"])

# ============================================================
# Demo tab
# ============================================================
with tab_demo:
    if is_sum:
        st.subheader(f"Summarize {dataset_label} documents")
        st.caption(
            "Run all three systems live and compare the generated summaries. "
            "G-Eval scores are only available in batch evaluation (Results tab)."
        )
    else:
        st.subheader(f"Ask about {dataset_label} documents")

    # Example questions / topics
    examples = _load_example_questions(dataset)
    if examples:
        label = "**Example topics** *(click to populate)*" if is_sum else "**Example questions** *(click to populate)*"
        st.markdown(label)
        ex_cols = st.columns(min(len(examples), 3))
        for i, (col, q) in enumerate(zip(ex_cols, examples[:3])):
            if col.button(
                q[:70] + ("…" if len(q) > 70 else ""),
                key=f"ex_{i}",
                use_container_width=True,
            ):
                st.session_state["chosen_example"] = q
                st.rerun()

        if len(examples) > 3:
            ex_cols2 = st.columns(min(len(examples) - 3, 3))
            for i, (col, q) in enumerate(zip(ex_cols2, examples[3:])):
                if col.button(
                    q[:70] + ("…" if len(q) > 70 else ""),
                    key=f"ex2_{i}",
                    use_container_width=True,
                ):
                    st.session_state["chosen_example"] = q
                    st.rerun()

    default_q = st.session_state.get("chosen_example", "")
    placeholder = (
        "e.g. Summarize the key highlights from this earnings call."
        if is_sum
        else "e.g. What is 3M's FY2018 capital expenditure in USD millions?"
    )
    question = st.text_area(
        "Your question:" if not is_sum else "Summarization task:",
        value=default_q,
        height=90,
        placeholder=placeholder,
        key="question_input",
    )

    btn_label = "▶ Summarize" if is_sum else "▶ Run A-RAG"
    run_btn = st.button(btn_label, type="primary", disabled=not question.strip())

    if run_btn and question.strip():
        st.session_state.pop("chosen_example", None)

        # ----------------------------------------------------------------
        # SUMMARIZATION demo (3 systems)
        # ----------------------------------------------------------------
        if is_sum:
            agent, config, sem_tool, chunk_tool_naive, all_chunks, client = load_sum_resources(dataset)

            if show_baseline:
                col_arag, col_naive, col_stuff = st.columns(3)
            else:
                col_arag = st.container()
                col_naive = None
                col_stuff = None

            # ---- A-RAG ----
            with col_arag:
                st.markdown("### 🤖 A-RAG")
                with st.spinner("Agent searching and reasoning…"):
                    t0 = time.time()
                    q_dict = {"id": "live", "question": question, "answer": ""}
                    arag_result = agent.run(question)
                    elapsed_a = time.time() - t0

                if arag_result.get("max_loops_reached"):
                    st.warning("⏱️ Hit the loop limit — summary may be partial.")

                st.markdown("**Summary:**")
                st.markdown(arag_result["answer"])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Loops",   arag_result["loops"])
                m2.metric("Words",   arag_result.get("word_count", len(arag_result["answer"].split())))
                m3.metric("Cost",    f"${arag_result.get('cost_usd', 0):.4f}")
                m4.metric("Time",    f"{elapsed_a:.1f}s")

                if arag_result.get("trace"):
                    st.markdown("---")
                    with st.expander("Retrieval trace & section coverage", expanded=False):
                        render_section_coverage(arag_result["trace"])
                        st.markdown("---")
                        render_trace(arag_result["trace"])

            # ---- Naive RAG ----
            if show_baseline and col_naive is not None:
                with col_naive:
                    st.markdown("### ⚡ Naive RAG")
                    with st.spinner("Top-k retrieval + single LLM call…"):
                        t0 = time.time()
                        q_dict = {"id": "live", "question": question, "answer": ""}
                        naive_result = run_naive_rag_summary(q_dict, config, sem_tool, chunk_tool_naive, client)
                        elapsed_n = time.time() - t0

                    st.markdown("**Summary:**")
                    st.info(naive_result["predicted"])

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Loops",  1)
                    m2.metric("Words",  naive_result.get("word_count", 0))
                    m3.metric("Cost",   f"${naive_result.get('cost_usd', 0):.4f}")
                    m4.metric("Time",   f"{elapsed_n:.1f}s")

            # ---- Long-Context Stuffing ----
            if show_baseline and col_stuff is not None:
                with col_stuff:
                    st.markdown("### 📚 Stuffing")
                    with st.spinner("Loading all chunks into context…"):
                        t0 = time.time()
                        q_dict = {"id": "live", "question": question, "answer": ""}
                        stuff_result = run_stuffing(q_dict, config, all_chunks, client, token_limit=180_000)
                        elapsed_s = time.time() - t0

                    if stuff_result.get("skipped"):
                        st.warning(f"⏭️ Skipped — {stuff_result.get('skip_reason', 'document too large')}")
                    else:
                        st.markdown("**Summary:**")
                        st.success(stuff_result["predicted"])

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Loops",  1)
                        m2.metric("Words",  stuff_result.get("word_count", 0))
                        m3.metric("Cost",   f"${stuff_result.get('cost_usd', 0):.4f}")
                        m4.metric("Time",   f"{elapsed_s:.1f}s")
                        st.caption(f"Chunks stuffed: {stuff_result.get('doc_chunks_count', 0)}")

        # ----------------------------------------------------------------
        # QA demo (2 systems: A-RAG + optional Naive RAG)
        # ----------------------------------------------------------------
        else:
            agent, config = load_agent(dataset)

            if show_baseline:
                col_arag, col_naive = st.columns(2)
            else:
                col_arag = st.container()
                col_naive = None

            # ---- A-RAG ----
            with col_arag:
                st.markdown("### 🤖 A-RAG (agentic)")
                with st.spinner("Agent searching and reasoning…"):
                    t0 = time.time()
                    result = agent.run(question)
                    elapsed = time.time() - t0

                if result.get("max_loops_reached"):
                    st.warning("⏱️ Hit the loop limit — answer may be partial.")

                st.markdown("**Answer:**")
                st.markdown(result["answer"])

                m1, m2, m3 = st.columns(3)
                m1.metric("Loops",  result["loops"])
                m2.metric("Tokens", f"{result['input_tokens'] + result['output_tokens']:,}")
                m3.metric("Time",   f"{elapsed:.1f}s")

                if result["trace"]:
                    st.markdown("---")
                    st.markdown("**Retrieval trace:**")
                    render_trace(result["trace"])

            # ---- Naive RAG ----
            if show_baseline and col_naive is not None:
                with col_naive:
                    st.markdown("### ⚡ Naive RAG (single-shot)")
                    with st.spinner("Single-shot retrieval…"):
                        t0 = time.time()
                        naive_result = run_naive_rag(question, config)
                        elapsed_naive = time.time() - t0

                    st.markdown("**Answer:**")
                    st.info(naive_result["answer"])

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Loops",  1)
                    m2.metric("Tokens", f"{naive_result['input_tokens'] + naive_result['output_tokens']:,}")
                    m3.metric("Time",   f"{elapsed_naive:.1f}s")

                    st.markdown(
                        f"**Retrieved chunk IDs:** `{naive_result['retrieved_chunk_ids']}`"
                    )


# ============================================================
# Results tab
# ============================================================
with tab_results:
    if is_sum:
        render_summarization_results_tab(dataset)
    else:
        render_results_tab(dataset)
