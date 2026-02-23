"""
Unit tests for AgentConfig loading.
"""
from __future__ import annotations

import pytest

from src.arag.config import AgentConfig


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.max_loops == 15
        assert cfg.top_k == 5
        assert cfg.temperature == 0.0

    def test_from_yaml_financebench(self):
        cfg = AgentConfig.from_yaml("configs/financebench.yaml")
        assert cfg.chunks_file == "data/financebench/chunks.json"
        assert cfg.index_dir == "data/financebench/index"
        assert cfg.embedding_model == "all-MiniLM-L6-v2"

    def test_from_yaml_finqa(self):
        cfg = AgentConfig.from_yaml("configs/finqa.yaml")
        assert "finqa" in cfg.chunks_file

    def test_from_yaml_finder(self):
        cfg = AgentConfig.from_yaml("configs/finder.yaml")
        assert "finder" in cfg.chunks_file

    def test_from_yaml_missing_file_raises(self):
        with pytest.raises(Exception):
            AgentConfig.from_yaml("configs/nonexistent_dataset.yaml")
