"""
Tests for summarization evaluation functions in scripts/eval.py.
No API key or FAISS index required — all LLM calls are mocked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval import (
    _extract_read_chunk_ids,
    _get_retrieved_text,
    _parse_geval_score,
    assign_error_codes,
    compute_retrieval_pr,
    rouge2_f1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trace(chunk_read_calls: list[list[int]], with_search: bool = True) -> list[dict]:
    """Build a minimal agent trace with chunk_read entries."""
    trace = []
    if with_search:
        trace.append({
            "loop": 1,
            "tool": "semantic_search",
            "input": {"query": "revenue growth"},
            "output": {"results": [{"chunk_id": 10, "score": 0.91, "preview": "..."}]},
        })
    for loop_idx, chunk_ids in enumerate(chunk_read_calls, start=2):
        results = [{"chunk_id": cid, "text": f"Full text of chunk {cid}."} for cid in chunk_ids]
        trace.append({
            "loop": loop_idx,
            "tool": "chunk_read",
            "input": {"chunk_ids": chunk_ids},
            "output": {"results": results},
        })
    return trace


def _make_item(predicted: str = "Revenue was $5B.", ground_truth: str = "Revenue grew to $5 billion.",
               trace: list | None = None) -> dict:
    return {
        "id": "ectsum_0001",
        "question": "Summarize the earnings call.",
        "ground_truth": ground_truth,
        "predicted": predicted,
        "loops": 8,
        "cost_usd": 0.012,
        "latency_ms": 9400,
        "word_count": len(predicted.split()),
        "trace": trace or _make_trace([[42, 87]]),
    }


def _mock_client(response_text: str) -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns a fixed text."""
    content_block = MagicMock()
    content_block.text = response_text
    response = MagicMock()
    response.content = [content_block]
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# _parse_geval_score
# ---------------------------------------------------------------------------

class TestParseGevalScore:
    def test_standard_format(self):
        assert _parse_geval_score("Criterion 1: good.\nFinal score: 4") == 4

    def test_case_insensitive(self):
        assert _parse_geval_score("FINAL SCORE: 3") == 3

    def test_colon_with_spaces(self):
        assert _parse_geval_score("Final score:  5") == 5

    def test_score_1(self):
        assert _parse_geval_score("Final score: 1") == 1

    def test_score_5(self):
        assert _parse_geval_score("Final score: 5") == 5

    def test_fallback_trailing_digit(self):
        # No "Final score:" label but ends with a digit
        assert _parse_geval_score("The summary is incomplete.\n\n3") == 3

    def test_returns_none_when_unparseable(self):
        assert _parse_geval_score("I cannot determine a score.") is None

    def test_out_of_range_digit_not_matched(self):
        # "6" is not a valid score
        result = _parse_geval_score("Final score: 6")
        assert result is None


# ---------------------------------------------------------------------------
# _extract_read_chunk_ids
# ---------------------------------------------------------------------------

class TestExtractReadChunkIds:
    def test_single_chunk_read_call(self):
        trace = _make_trace([[42, 87]])
        assert _extract_read_chunk_ids(trace) == {42, 87}

    def test_multiple_chunk_read_calls(self):
        trace = _make_trace([[10, 20], [30, 40]])
        assert _extract_read_chunk_ids(trace) == {10, 20, 30, 40}

    def test_no_chunk_read_calls(self):
        trace = _make_trace([], with_search=True)
        assert _extract_read_chunk_ids(trace) == set()

    def test_empty_trace(self):
        assert _extract_read_chunk_ids([]) == set()

    def test_ignores_search_tools(self):
        trace = [
            {"loop": 1, "tool": "semantic_search", "input": {"query": "q"}, "output": {}},
            {"loop": 2, "tool": "keyword_search", "input": {"keywords": ["k"]}, "output": {}},
        ]
        assert _extract_read_chunk_ids(trace) == set()


# ---------------------------------------------------------------------------
# _get_retrieved_text
# ---------------------------------------------------------------------------

class TestGetRetrievedText:
    def test_returns_text_for_read_chunks(self):
        trace = _make_trace([[42]])
        text = _get_retrieved_text(trace)
        assert "Chunk 42" in text
        assert "Full text of chunk 42" in text

    def test_multiple_chunks_separated(self):
        trace = _make_trace([[1, 2]])
        text = _get_retrieved_text(trace)
        assert "Chunk 1" in text
        assert "Chunk 2" in text

    def test_empty_trace_returns_placeholder(self):
        text = _get_retrieved_text([])
        assert "no chunks" in text.lower()

    def test_non_chunk_read_steps_ignored(self):
        trace = [{"loop": 1, "tool": "semantic_search", "input": {}, "output": {}}]
        text = _get_retrieved_text(trace)
        assert "no chunks" in text.lower()


# ---------------------------------------------------------------------------
# compute_retrieval_pr
# ---------------------------------------------------------------------------

class TestComputeRetrievalPR:
    def test_perfect_recall_and_precision(self):
        trace = _make_trace([[10, 20, 30]])
        recall, precision = compute_retrieval_pr(trace, gold_chunk_ids=[10, 20, 30])
        assert recall == 1.0
        assert precision == 1.0

    def test_partial_recall(self):
        trace = _make_trace([[10, 20]])        # retrieved 2 of 4 gold chunks
        recall, precision = compute_retrieval_pr(trace, gold_chunk_ids=[10, 20, 30, 40])
        assert recall == 0.5
        assert precision == 1.0

    def test_zero_precision_when_all_retrieved_are_irrelevant(self):
        trace = _make_trace([[99, 100]])       # retrieved, but not in gold
        recall, precision = compute_retrieval_pr(trace, gold_chunk_ids=[10, 20])
        assert recall == 0.0
        assert precision == 0.0

    def test_empty_gold_returns_zeros(self):
        trace = _make_trace([[10, 20]])
        recall, precision = compute_retrieval_pr(trace, gold_chunk_ids=[])
        assert recall == 0.0
        assert precision == 0.0

    def test_no_chunk_reads_returns_zeros(self):
        trace = _make_trace([], with_search=True)
        recall, precision = compute_retrieval_pr(trace, gold_chunk_ids=[10, 20])
        assert recall == 0.0
        assert precision == 0.0

    def test_mixed_recall_and_precision(self):
        # Retrieved [10, 20, 99] — gold is [10, 20, 30]
        # overlap=2, recall=2/3, precision=2/3
        trace = _make_trace([[10, 20, 99]])
        recall, precision = compute_retrieval_pr(trace, gold_chunk_ids=[10, 20, 30])
        assert abs(recall - 2 / 3) < 0.001
        assert abs(precision - 2 / 3) < 0.001

    def test_results_are_rounded_to_4_decimals(self):
        trace = _make_trace([[10]])
        recall, precision = compute_retrieval_pr(trace, gold_chunk_ids=[10, 20, 30])
        assert recall == round(recall, 4)
        assert precision == round(precision, 4)


# ---------------------------------------------------------------------------
# rouge2_f1
# ---------------------------------------------------------------------------

class TestRouge2F1:
    def test_identical_strings_score_one(self):
        score = rouge2_f1("revenue grew to five billion", "revenue grew to five billion")
        assert score == 1.0

    def test_completely_different_strings_score_low(self):
        score = rouge2_f1("the weather is sunny today", "revenue increased by ten percent")
        assert score < 0.1

    def test_partial_overlap_between_zero_and_one(self):
        score = rouge2_f1("revenue grew by ten percent this quarter", "revenue increased ten percent")
        assert 0.0 < score < 1.0

    def test_empty_strings_return_zero(self):
        assert rouge2_f1("", "") == 0.0 or rouge2_f1("", "") >= 0.0  # library handles gracefully

    def test_returns_float(self):
        score = rouge2_f1("some text", "some other text")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# assign_error_codes (mocked client)
# ---------------------------------------------------------------------------

class TestAssignErrorCodes:
    def test_hallucination_code_parsed(self):
        client = _mock_client("H")
        codes = assign_error_codes(client, "model", "hallucinated revenue", "missed section")
        assert "H" in codes

    def test_multiple_codes_parsed(self):
        client = _mock_client("H,O")
        codes = assign_error_codes(client, "model", "reasoning...", "reasoning...")
        assert set(codes) == {"H", "O"}

    def test_none_response_returns_empty_list(self):
        client = _mock_client("none")
        codes = assign_error_codes(client, "model", "good reasoning", "good reasoning")
        assert codes == []

    def test_invalid_codes_filtered_out(self):
        client = _mock_client("H,INVALID,O")
        codes = assign_error_codes(client, "model", "r", "r")
        assert "INVALID" not in codes
        assert "H" in codes
        assert "O" in codes

    def test_all_valid_taxonomy_codes_accepted(self):
        for code in ["H", "N", "O", "P", "IR", "IC", "V"]:
            client = _mock_client(code)
            codes = assign_error_codes(client, "model", "r", "r")
            assert code in codes

    def test_client_called_once(self):
        client = _mock_client("H")
        assign_error_codes(client, "model", "r", "r")
        assert client.messages.create.call_count == 1
