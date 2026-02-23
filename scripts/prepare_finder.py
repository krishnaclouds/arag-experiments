#!/usr/bin/env python3
"""
Prepare FinDER for A-RAG evaluation.

FinDER (Financial Dataset for Expert Retrieval) contains 5,703 expert-annotated
query–evidence–answer triplets from 10-K filings (ICLR 2025).

The corpus is built from all unique reference passages across all queries
(deduplicated). This gives a realistic retrieval challenge: each question's
gold evidence is mixed with ~thousands of passages from other queries.

Usage:
    uv run python scripts/prepare_finder.py
    uv run python scripts/prepare_finder.py --output data/finder --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.indexing.build_index import build_faiss_index


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare FinDER for A-RAG evaluation")
    p.add_argument("--output", default="data/finder", help="Output directory")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit questions.json to first N rows (for quick experiments)",
    )
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load FinDER ----
    print("Loading FinDER from HuggingFace (Linq-AI-Research/FinDER) …")
    from datasets import load_dataset

    ds = load_dataset("Linq-AI-Research/FinDER", split="train")
    rows = list(ds)
    print(f"Loaded {len(rows)} rows")

    # ---- Build corpus from unique reference passages ----
    print("\nBuilding corpus from unique reference passages …")
    seen_passages: dict[str, int] = {}  # passage_text → chunk_id
    chunks: list[str] = []

    for row in rows:
        for passage in row.get("references") or []:
            passage = passage.strip()
            if not passage or passage in seen_passages:
                continue
            chunk_id = len(chunks)
            seen_passages[passage] = chunk_id
            chunks.append(f"{chunk_id}:{passage}")

    print(f"Unique passages (chunks): {len(chunks)}")

    chunks_file = str(output_dir / "chunks.json")
    with open(chunks_file, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Wrote {chunks_file}")

    # ---- Build FAISS index ----
    index_dir = str(output_dir / "index")
    print(f"\nBuilding FAISS index …")
    build_faiss_index(
        chunks_file=chunks_file,
        output_dir=index_dir,
        embedding_model=args.embedding_model,
        device=args.device,
    )

    # ---- Build questions.json ----
    print("\nBuilding questions.json …")
    eval_rows = rows[: args.limit] if args.limit else rows

    questions: list[dict] = []
    for row in eval_rows:
        query = (row.get("text") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not query or not answer:
            continue

        # Evidence: join all reference passages for this query
        refs = row.get("references") or []
        evidence = " | ".join(r.strip() for r in refs if r.strip())

        # Map reference passages to chunk IDs
        evidence_chunk_ids = [
            seen_passages[r.strip()]
            for r in refs
            if r.strip() in seen_passages
        ]

        questions.append(
            {
                "id": row.get("_id", ""),
                "source": "finder",
                "question": query,
                "answer": answer,
                "question_type": row.get("type", ""),
                "evidence": evidence,
                "evidence_relations": [],
                "category": row.get("category", ""),
                "reasoning_required": row.get("reasoning", False),
                "gold_chunk_ids": evidence_chunk_ids,
            }
        )

    questions_file = str(output_dir / "questions.json")
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Wrote {len(questions)} questions → {questions_file}")

    print("\nDone. Next steps:")
    print(f"  uv run python scripts/batch_runner.py --config configs/finder.yaml --output results/finder --workers 3")


if __name__ == "__main__":
    main()
