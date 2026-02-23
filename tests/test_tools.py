"""
Unit tests for the three A-RAG retrieval tools.
Uses a small in-memory corpus — no FAISS index or API key required.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.arag.tools.keyword_search import KeywordSearchTool
from src.arag.tools.chunk_read import ChunkReadTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    "0:[COMPANY: 3M | FILING: 10-K 2018] Capital expenditures were 1577 million dollars in fiscal year 2018.",
    "1:[COMPANY: 3M | FILING: 10-K 2018] Revenue increased to 32.8 billion in 2018 from 31.7 billion in 2017.",
    "2:[COMPANY: AMAZON | FILING: 10-K 2019] Operating income for Amazon was 14.5 billion in fiscal 2019.",
    "3:[COMPANY: AMAZON | FILING: 10-K 2019] Net sales increased 20% year over year driven by AWS growth.",
    "4:[COMPANY: APPLE | FILING: 10-K 2023] Gross margin was 44.1 percent in fiscal year 2023.",
]


@pytest.fixture()
def chunks_file(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(SAMPLE_CHUNKS))
    return str(path)


# ---------------------------------------------------------------------------
# KeywordSearchTool
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_single_keyword_returns_results(self, chunks_file):
        tool = KeywordSearchTool(chunks_file=chunks_file)
        result = tool.run(keywords=["capital expenditure"], top_k=3)
        assert "results" in result
        assert len(result["results"]) >= 1
        assert result["results"][0]["chunk_id"] == 0

    def test_multi_keyword_scoring(self, chunks_file):
        tool = KeywordSearchTool(chunks_file=chunks_file)
        result = tool.run(keywords=["revenue", "2018"], top_k=3)
        ids = [r["chunk_id"] for r in result["results"]]
        # chunk 1 contains both keywords → should rank high
        assert 1 in ids

    def test_top_k_respected(self, chunks_file):
        tool = KeywordSearchTool(chunks_file=chunks_file)
        result = tool.run(keywords=["2018"], top_k=2)
        assert len(result["results"]) <= 2

    def test_no_match_returns_empty(self, chunks_file):
        tool = KeywordSearchTool(chunks_file=chunks_file)
        result = tool.run(keywords=["nonexistentxyzabc123"], top_k=5)
        assert result["results"] == []

    def test_schema_is_valid_anthropic_tool(self, chunks_file):
        tool = KeywordSearchTool(chunks_file=chunks_file)
        schema = tool.schema
        assert schema["name"] == "keyword_search"
        assert "input_schema" in schema
        assert "keywords" in schema["input_schema"]["properties"]

    def test_longer_keyword_scores_higher(self, chunks_file):
        """Longer matching keyword = higher score per the paper formula."""
        tool = KeywordSearchTool(chunks_file=chunks_file)
        r_short = tool.run(keywords=["2018"], top_k=5)
        r_long = tool.run(keywords=["capital expenditures"], top_k=5)
        score_short = next((r["score"] for r in r_short["results"] if r["chunk_id"] == 0), 0)
        score_long = next((r["score"] for r in r_long["results"] if r["chunk_id"] == 0), 0)
        assert score_long > score_short


# ---------------------------------------------------------------------------
# ChunkReadTool
# ---------------------------------------------------------------------------

class TestChunkRead:
    def test_read_single_chunk(self, chunks_file):
        tool = ChunkReadTool(chunks_file=chunks_file)
        result = tool.run(chunk_ids=[0])
        assert len(result["results"]) == 1
        assert "1577" in result["results"][0]["text"]

    def test_read_multiple_chunks(self, chunks_file):
        tool = ChunkReadTool(chunks_file=chunks_file)
        result = tool.run(chunk_ids=[0, 2, 4])
        assert len(result["results"]) == 3
        ids_returned = {r["chunk_id"] for r in result["results"]}
        assert ids_returned == {0, 2, 4}

    def test_dedup_set_skips_already_read(self, chunks_file):
        tool = ChunkReadTool(chunks_file=chunks_file)
        tool.run(chunk_ids=[1])
        result2 = tool.run(chunk_ids=[1])
        # Second read of the same chunk: the C_read dedup set suppresses it —
        # the result is either empty or carries a skip/already-read marker.
        results = result2["results"]
        if results:
            # If returned, it must not have full text (it's a skip notice)
            assert not any(r.get("chunk_id") == 1 and "text" in r and len(r["text"]) > 10 for r in results)
        # else: empty list is also acceptable — chunk was silently skipped

    def test_reset_clears_dedup_set(self, chunks_file):
        tool = ChunkReadTool(chunks_file=chunks_file)
        tool.run(chunk_ids=[1])
        tool.reset()
        result = tool.run(chunk_ids=[1])
        # After reset, chunk 1 should be readable again
        assert any(r.get("chunk_id") == 1 and "text" in r for r in result["results"])

    def test_invalid_chunk_id_handled(self, chunks_file):
        tool = ChunkReadTool(chunks_file=chunks_file)
        result = tool.run(chunk_ids=[9999])
        # Should not raise; should return an error entry or empty
        assert isinstance(result, dict)

    def test_schema_is_valid_anthropic_tool(self, chunks_file):
        tool = ChunkReadTool(chunks_file=chunks_file)
        schema = tool.schema
        assert schema["name"] == "chunk_read"
        assert "chunk_ids" in schema["input_schema"]["properties"]
