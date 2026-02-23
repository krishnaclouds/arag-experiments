"""
Unit tests for the text chunker.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.arag.indexing.chunker import chunk_text, build_chunks_json


class TestChunkText:
    def test_short_text_is_single_chunk(self):
        chunks = chunk_text("Hello world.", max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_splits_into_multiple_chunks(self):
        # ~400 words × ~5 chars/word = ~2000 chars ≈ 500 tokens
        sentence = "The revenue increased significantly this quarter. "
        long_text = sentence * 60  # ~3000 tokens worth
        chunks = chunk_text(long_text, max_tokens=500)
        assert len(chunks) > 1

    def test_chunks_cover_all_content(self):
        sentence = "Revenue was strong this year. "
        long_text = sentence * 40
        chunks = chunk_text(long_text, max_tokens=200)
        # All words should appear across the chunks
        combined = " ".join(chunks)
        assert "Revenue" in combined
        assert "strong" in combined

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", max_tokens=500)
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = chunk_text("   \n\n   ", max_tokens=500)
        assert chunks == []


class TestBuildChunksJson:
    def test_produces_valid_json_file(self, tmp_path):
        docs = [
            {"text": "Apple revenue was 100 billion in FY2023.", "metadata": "[COMPANY: Apple]"},
        ]
        out = str(tmp_path / "chunks.json")
        build_chunks_json(docs, out, max_tokens=500)
        data = json.loads(Path(out).read_text())
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_chunk_format_has_id_prefix(self, tmp_path):
        docs = [{"text": "Short text.", "metadata": "[META]"}]
        out = str(tmp_path / "chunks.json")
        build_chunks_json(docs, out, max_tokens=500)
        data = json.loads(Path(out).read_text())
        # Each entry should start with "N:" where N is an integer
        for entry in data:
            parts = entry.split(":", 1)
            assert parts[0].isdigit()

    def test_metadata_prepended_to_chunk(self, tmp_path):
        docs = [{"text": "Revenue was 500M.", "metadata": "[COMPANY: TestCo]"}]
        out = str(tmp_path / "chunks.json")
        build_chunks_json(docs, out, max_tokens=500)
        data = json.loads(Path(out).read_text())
        assert any("TestCo" in entry for entry in data)

    def test_multiple_documents(self, tmp_path):
        docs = [
            {"text": "Doc one content.", "metadata": "[DOC: 1]"},
            {"text": "Doc two content.", "metadata": "[DOC: 2]"},
        ]
        out = str(tmp_path / "chunks.json")
        build_chunks_json(docs, out, max_tokens=500)
        data = json.loads(Path(out).read_text())
        assert len(data) >= 2
