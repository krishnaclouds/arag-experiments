from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class SemanticSearchTool:
    """
    Dense vector search over sentence-level embeddings.

    Sentence embeddings are stored in a FAISS flat inner-product index
    (equivalent to cosine similarity when vectors are L2-normalised).
    Results are aggregated to chunk level by taking the max sentence score
    per chunk, then the top-k chunks are returned.
    """

    name = "semantic_search"
    description = (
        "Search for chunks semantically related to a natural-language query using "
        "dense vector similarity. Best when you need conceptually related content "
        "and don't know the exact wording (e.g. 'factors affecting profitability', "
        "'management outlook for next year')."
    )

    def __init__(
        self,
        chunks_file: str,
        index_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        # Load chunks
        with open(chunks_file) as f:
            raw: list[str] = json.load(f)
        self._chunks: dict[int, str] = {}
        for entry in raw:
            idx_str, _, text = entry.partition(":")
            self._chunks[int(idx_str)] = text

        # Load FAISS index and sentence→chunk mapping
        index_path = Path(index_dir)
        self._index = faiss.read_index(str(index_path / "sentences.faiss"))
        with open(index_path / "sentence_map.json") as f:
            # sentence_map[i] = chunk_id for the i-th sentence embedding
            self._sentence_map: list[int] = json.load(f)

        # Load embedding model (lazy import to avoid slow startup when unused)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(embedding_model, device=device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str, top_k: int = 5) -> dict[str, Any]:
        # Embed query
        query_vec = self._model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Search more sentences than needed, then aggregate by chunk
        k_search = min(top_k * 10, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k_search)

        # Max-pool sentence scores per chunk
        chunk_scores: dict[int, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_id = self._sentence_map[idx]
            if chunk_id not in chunk_scores or score > chunk_scores[chunk_id]:
                chunk_scores[chunk_id] = float(score)

        top_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        return {
            "results": [
                {
                    "chunk_id": chunk_id,
                    "score": round(score, 4),
                    "preview": (
                        self._chunks[chunk_id][:300] + "…"
                        if len(self._chunks[chunk_id]) > 300
                        else self._chunks[chunk_id]
                    ),
                }
                for chunk_id, score in top_chunks
            ]
        }

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query to search for semantically similar content.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }
