from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    max_tokens: int = 1000,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Split *text* into chunks of at most *max_tokens* tokens (approximated as
    chars / 4), aligned to sentence boundaries. The last *overlap_sentences*
    sentences of each chunk are carried into the next for context continuity.
    """
    sentences = split_sentences(text)
    chars_limit = max_tokens * 4  # ~4 chars per token

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chars_limit and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# chunks.json builder
# ---------------------------------------------------------------------------

def build_chunks_json(
    documents: list[dict],
    output_file: str,
    max_tokens: int = 1000,
) -> list[str]:
    """
    Build a chunks.json file from a list of documents.

    Each document dict must have:
      - "text"     : str  — the document text
      - "metadata" : str  — prefix added to every chunk
                            e.g. "[COMPANY: Apple | FILING: 10-K FY2023 | SECTION: Item 7]"

    Returns the list of "id:text" strings written to *output_file*.
    """
    all_chunks: list[str] = []
    chunk_id = 0

    for doc in documents:
        text = doc["text"]
        prefix = doc.get("metadata", "")

        for chunk_body in chunk_text(text, max_tokens=max_tokens):
            full = f"{prefix} {chunk_body}".strip() if prefix else chunk_body
            all_chunks.append(f"{chunk_id}:{full}")
            chunk_id += 1

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Wrote {len(all_chunks)} chunks → {output_file}")
    return all_chunks
