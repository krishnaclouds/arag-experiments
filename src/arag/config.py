from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


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
