from __future__ import annotations

import json
from typing import Any


class ChunkReadTool:
    """
    Return full text of chunks by ID.

    Maintains a C_read set (per agent session) so already-read chunks are
    skipped — preventing redundant token consumption as described in the
    A-RAG paper (arXiv:2602.03442).
    """

    name = "chunk_read"
    description = (
        "Read the complete text of one or more chunks by their IDs. "
        "Use after keyword_search or semantic_search to inspect the full content "
        "of promising chunks. Already-read chunks are automatically skipped."
    )

    def __init__(self, chunks_file: str) -> None:
        with open(chunks_file) as f:
            raw: list[str] = json.load(f)
        self._chunks: dict[int, str] = {}
        for entry in raw:
            idx_str, _, text = entry.partition(":")
            self._chunks[int(idx_str)] = text

        self._read: set[int] = set()  # C_read set

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, chunk_ids: list[int]) -> dict[str, Any]:
        results = []
        skipped = []

        for chunk_id in chunk_ids:
            if chunk_id in self._read:
                skipped.append(chunk_id)
                continue
            if chunk_id not in self._chunks:
                results.append({"chunk_id": chunk_id, "error": "chunk not found"})
                continue
            self._read.add(chunk_id)
            results.append({"chunk_id": chunk_id, "text": self._chunks[chunk_id]})

        response: dict[str, Any] = {"results": results}
        if skipped:
            response["skipped"] = skipped
            response["note"] = (
                f"Chunks {skipped} were already read this session and skipped."
            )
        return response

    def reset(self) -> None:
        """Clear C_read — call at the start of each new question."""
        self._read.clear()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "List of chunk IDs to read in full "
                            "(obtained from keyword_search or semantic_search results)."
                        ),
                    }
                },
                "required": ["chunk_ids"],
            },
        }
