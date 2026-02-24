"""
Tests for prompt selection and G-Eval prompt templates.
No API key or index required.
"""
from __future__ import annotations

import pytest

from src.arag.agent.prompts import (
    ERROR_CODE_PROMPT,
    GEVAL_CRITERIA_PROMPT,
    GEVAL_SCORE_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from src.arag.agent.loop import _PROMPTS
from src.arag.config import AgentConfig, PRICING, compute_cost


# ---------------------------------------------------------------------------
# Prompt dispatch
# ---------------------------------------------------------------------------

class TestPromptDispatch:
    def test_qa_task_uses_qa_prompt(self):
        cfg = AgentConfig(task_type="qa")
        assert _PROMPTS.get(cfg.task_type) is SYSTEM_PROMPT

    def test_summarization_task_uses_summarization_prompt(self):
        cfg = AgentConfig(task_type="summarization")
        assert _PROMPTS.get(cfg.task_type) is SUMMARIZATION_SYSTEM_PROMPT

    def test_unknown_task_type_falls_back_to_qa_prompt(self):
        # _PROMPTS.get with unknown key returns None; loop.py falls back to SYSTEM_PROMPT
        assert _PROMPTS.get("unknown_task_type", SYSTEM_PROMPT) is SYSTEM_PROMPT

    def test_qa_and_summarization_prompts_are_different(self):
        assert SYSTEM_PROMPT != SUMMARIZATION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Summarization prompt content
# ---------------------------------------------------------------------------

class TestSummarizationPrompt:
    def test_contains_broad_retrieval_instruction(self):
        assert "broadly" in SUMMARIZATION_SYSTEM_PROMPT.lower()

    def test_contains_anti_hallucination_instruction(self):
        # Must instruct agent to only include supported claims
        assert "retrieved" in SUMMARIZATION_SYSTEM_PROMPT.lower()
        assert "do not" in SUMMARIZATION_SYSTEM_PROMPT.lower() or "only" in SUMMARIZATION_SYSTEM_PROMPT.lower()

    def test_contains_synthesis_rules_section(self):
        assert "Synthesis rules" in SUMMARIZATION_SYSTEM_PROMPT

    def test_does_not_mention_stop_early(self):
        # The QA prompt tells the agent to stop when it has enough; the
        # summarization prompt should NOT have that instruction
        assert "sufficient evidence" not in SUMMARIZATION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# G-Eval prompt templates
# ---------------------------------------------------------------------------

class TestGEvalPrompts:
    def test_criteria_prompt_contains_dimension_placeholder(self):
        assert "{dimension}" in GEVAL_CRITERIA_PROMPT

    def test_score_prompt_contains_all_placeholders(self):
        for placeholder in ["{dimension}", "{criteria}", "{retrieved_chunks}", "{reference}", "{predicted}"]:
            assert placeholder in GEVAL_SCORE_PROMPT

    def test_score_prompt_requests_final_score_format(self):
        assert "Final score:" in GEVAL_SCORE_PROMPT

    def test_error_code_prompt_contains_all_taxonomy_codes(self):
        for code in ["H", "N", "O", "P", "IR", "IC", "V"]:
            assert code in ERROR_CODE_PROMPT

    def test_error_code_prompt_has_reasoning_placeholders(self):
        assert "{faithfulness_reasoning}" in ERROR_CODE_PROMPT
        assert "{coverage_reasoning}" in ERROR_CODE_PROMPT

    def test_criteria_prompt_renders(self):
        rendered = GEVAL_CRITERIA_PROMPT.format(dimension="faithfulness")
        assert "faithfulness" in rendered
        assert "{dimension}" not in rendered

    def test_score_prompt_renders(self):
        rendered = GEVAL_SCORE_PROMPT.format(
            dimension="coverage",
            criteria="1. Completeness ...",
            retrieved_chunks="[chunk text]",
            reference="reference summary",
            predicted="predicted summary",
        )
        assert "coverage" in rendered
        assert "{" not in rendered


# ---------------------------------------------------------------------------
# AgentConfig new fields
# ---------------------------------------------------------------------------

class TestAgentConfigNewFields:
    def test_defaults_task_type_is_qa(self):
        cfg = AgentConfig()
        assert cfg.task_type == "qa"

    def test_defaults_summarization_style(self):
        cfg = AgentConfig()
        assert cfg.summarization_style == "earnings_call"

    def test_defaults_target_length(self):
        cfg = AgentConfig()
        assert cfg.target_length == 200

    def test_summarization_config_overrides(self):
        cfg = AgentConfig(task_type="summarization", summarization_style="section", target_length=300)
        assert cfg.task_type == "summarization"
        assert cfg.summarization_style == "section"
        assert cfg.target_length == 300

    def test_existing_qa_config_unaffected(self):
        """Existing QA configs that don't set task_type still default to 'qa'."""
        cfg = AgentConfig.from_yaml("configs/financebench.yaml")
        assert cfg.task_type == "qa"
        assert cfg.max_loops == 15


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

class TestComputeCost:
    def test_sonnet_cost_calculation(self):
        # 1M input + 1M output for sonnet = $3.00 + $15.00 = $18.00
        cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
        assert abs(cost - 18.00) < 0.01

    def test_haiku_is_cheaper_than_sonnet(self):
        tokens = 100_000
        cost_haiku = compute_cost("claude-haiku-4-5-20251001", tokens, tokens)
        cost_sonnet = compute_cost("claude-sonnet-4-6", tokens, tokens)
        assert cost_haiku < cost_sonnet

    def test_unknown_model_falls_back_to_sonnet_pricing(self):
        cost_unknown = compute_cost("claude-unknown-model", 100_000, 10_000)
        cost_sonnet = compute_cost("claude-sonnet-4-6", 100_000, 10_000)
        assert cost_unknown == cost_sonnet

    def test_zero_tokens_returns_zero_cost(self):
        assert compute_cost("claude-sonnet-4-6", 0, 0) == 0.0

    def test_pricing_table_has_required_models(self):
        assert "claude-sonnet-4-6" in PRICING
        assert "claude-haiku-4-5-20251001" in PRICING
        for model, prices in PRICING.items():
            assert "input" in prices
            assert "output" in prices
            assert prices["output"] > prices["input"]  # output always more expensive
