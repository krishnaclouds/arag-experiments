from __future__ import annotations

import json
from typing import Any


class KeywordSearchTool:
    """
    Exact-match keyword search over the chunk corpus.

    Scoring formula from the A-RAG paper (arXiv:2602.03442):
        Score(chunk, keywords) = Σ count(k, chunk_text) × len(k)

    Returns top-k chunks by score with a keyword-highlighted snippet.
    """

    name = "keyword_search"
    description = (
        "Search for chunks containing specific keywords using exact text matching. "
        "Best for finding specific financial terms, company names, metric names, "
        "years, or numerical values (e.g. 'gross margin', 'fiscal 2023', 'EBITDA')."
    )

    def __init__(self, chunks_file: str) -> None:
        with open(chunks_file) as f:
            raw: list[str] = json.load(f)
        # Parse "id:text" format
        self._chunks: dict[int, str] = {}
        for entry in raw:
            idx_str, _, text = entry.partition(":")
            self._chunks[int(idx_str)] = text

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, keywords: list[str], top_k: int = 5) -> dict[str, Any]:
        scored: list[tuple[int, float, str]] = []
        for chunk_id, text in self._chunks.items():
            s = self._score(text, keywords)
            if s > 0:
                scored.append((chunk_id, s, text))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        return {
            "results": [
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "snippet": self._snippet(text, keywords),
                }
                for chunk_id, score, text in top
            ],
            "total_matched": len(scored),
        }

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of 1–3 word exact-match search terms "
                            "(e.g. ['gross margin', 'FY2023', 'operating income'])."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["keywords"],
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _score(text: str, keywords: list[str]) -> float:
        text_lower = text.lower()
        total = 0.0
        for kw in keywords:
            kw_lower = kw.lower()
            count = text_lower.count(kw_lower)
            total += count * len(kw)
        return total

    @staticmethod
    def _snippet(text: str, keywords: list[str], window: int = 250) -> str:
        """Extract a window of text centred on the first keyword match."""
        text_lower = text.lower()
        best_pos = len(text)
        for kw in keywords:
            pos = text_lower.find(kw.lower())
            if 0 <= pos < best_pos:
                best_pos = pos

        if best_pos == len(text):
            return text[:window]

        start = max(0, best_pos - window // 2)
        end = min(len(text), best_pos + window // 2)
        snippet = text[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return snippet
