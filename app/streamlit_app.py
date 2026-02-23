"""
A-RAG Financial Demo — Streamlit UI

Run with:
    uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.agent.loop import AgentLoop
from src.arag.config import AgentConfig, get_api_key
from src.arag.tools.chunk_read import ChunkReadTool
from src.arag.tools.keyword_search import KeywordSearchTool
from src.arag.tools.semantic_search import SemanticSearchTool
from baselines.naive_rag import run_naive_rag

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
}

# Real questions from each dataset (pulled from questions.json at load time)
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


def _load_example_questions(dataset: str) -> list[str]:
    try:
        path = f"data/{dataset}/questions.json"
        qs = json.load(open(path))
        # Pick 5 varied questions
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


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def render_trace(trace: list[dict]) -> None:
    """Render the agent's step-by-step retrieval trace."""
    if not trace:
        return

    # Summary bar
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


def render_results_tab(dataset: str) -> None:
    """Render the benchmark results tab."""
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

    # --- Headline metrics ---
    st.subheader("Benchmark Results")

    if naive_data:
        npreds, nevals = naive_data
        naive_correct = sum(1 for r in nevals if r.get("llm_correct"))
        naive_cm = sum(1 for r in nevals if r.get("contain_match"))
        naive_avg_tokens = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in npreds) / len(npreds)

        # Comparison table
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
    from collections import Counter
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

    # Dataset selector — only show datasets with data
    available = {k: v for k, v in DATASETS.items() if _dataset_ready(v)}
    if not available:
        st.error("No datasets ready. Run a prepare_*.py script first.")
        st.stop()

    dataset_label = st.selectbox("Dataset", list(available.keys()))
    dataset = available[dataset_label]

    st.divider()
    show_baseline = st.toggle("Compare with naive RAG", value=True)

    st.divider()
    st.markdown("**How A-RAG works**")
    st.markdown(
        "The agent iteratively calls three retrieval tools:\n\n"
        "- 🔑 **keyword_search** — exact match scoring\n"
        "- 🔍 **semantic_search** — FAISS cosine similarity\n"
        "- 📄 **chunk_read** — full text of specific chunks\n\n"
        "It loops until it has enough evidence to answer or hits the loop limit."
    )

    # Quick eval stats in sidebar if available
    arag_data = _load_eval_results(dataset)
    if arag_data:
        preds, evals = arag_data
        n = len(preds)
        correct = sum(1 for r in evals if r.get("llm_correct"))
        st.divider()
        st.markdown("**Benchmark (this dataset)**")
        st.metric("A-RAG LLM-Accuracy", f"{correct/n:.1%} ({correct}/{n})")
        naive_data = _load_naive_eval_results(dataset)
        if naive_data:
            npreds, nevals = naive_data
            nc = sum(1 for r in nevals if r.get("llm_correct"))
            st.metric("Naive RAG LLM-Accuracy", f"{nc/len(npreds):.1%} ({nc}/{len(npreds)})")


# ---------------------------------------------------------------------------
# Main content — tabs
# ---------------------------------------------------------------------------

tab_demo, tab_results = st.tabs(["💬 Ask a Question", "📊 Benchmark Results"])

# ---- Demo tab ----
with tab_demo:
    st.subheader(f"Ask about {dataset_label} documents")

    # Example questions
    examples = _load_example_questions(dataset)
    if examples:
        st.markdown("**Example questions** *(click to populate)*")
        ex_cols = st.columns(min(len(examples), 3))
        chosen_example = st.session_state.get("chosen_example", "")
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
    question = st.text_area(
        "Your question:",
        value=default_q,
        height=90,
        placeholder="e.g. What is 3M's FY2018 capital expenditure in USD millions?",
        key="question_input",
    )

    run_btn = st.button("▶ Run A-RAG", type="primary", disabled=not question.strip())

    if run_btn and question.strip():
        # Clear previous example so next run starts fresh
        st.session_state.pop("chosen_example", None)

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
            m1.metric("Loops", result["loops"])
            m2.metric("Tokens", f"{result['input_tokens'] + result['output_tokens']:,}")
            m3.metric("Time", f"{elapsed:.1f}s")

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
                m1.metric("Loops", 1)
                m2.metric("Tokens", f"{naive_result['input_tokens'] + naive_result['output_tokens']:,}")
                m3.metric("Time", f"{elapsed_naive:.1f}s")

                st.markdown(
                    f"**Retrieved chunk IDs:** `{naive_result['retrieved_chunk_ids']}`"
                )


# ---- Results tab ----
with tab_results:
    render_results_tab(dataset)
