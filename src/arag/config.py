from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Cost per million tokens (input, output). Update if Anthropic pricing changes.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD for a given model and token counts."""
    pricing = PRICING.get(model, PRICING["claude-sonnet-4-6"])
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


@dataclass
class AgentConfig:
    # LLM
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 16384

    # Agent loop
    max_loops: int = 15
    max_token_budget: int = 128000

    # Retrieval
    top_k: int = 5

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Task type — "qa" for all existing datasets; "summarization" for new summarization tasks
    task_type: str = "qa"
    summarization_style: str = "earnings_call"  # earnings_call | section | query_focused | comparison
    target_length: int = 200                     # approximate target word count for summarization output

    # Paths (must be set per dataset via YAML or constructor)
    chunks_file: str = ""
    index_dir: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> AgentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key
