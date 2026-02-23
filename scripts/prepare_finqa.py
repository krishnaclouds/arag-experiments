#!/usr/bin/env python3
"""
Prepare FinQA for A-RAG evaluation.

Downloads train/dev/test splits from the FinQA GitHub repo, builds a
shared corpus from ALL splits (so test questions can retrieve from it),
then writes chunks.json, a FAISS index, and questions.json (test split).

Usage:
    uv run python scripts/prepare_finqa.py
    uv run python scripts/prepare_finqa.py --output data/finqa --eval-split test
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.indexing.build_index import build_faiss_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINQA_BASE = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
SPLITS = ["train", "dev", "test"]


# ---------------------------------------------------------------------------
# Table linearisation (FinQA-specific format)
# ---------------------------------------------------------------------------

def _linearise_finqa_table(table: list[list]) -> str:
    """
    Convert a FinQA table (list of rows, each row a list of cell strings)
    into pipe-delimited Markdown.

    FinQA table rows look like:
      ['', 'Oct 2009', 'Nov 2008']   ← header row (first cell empty = row label)
      ['Revenue', '1900', '2000']
    """
    if not table:
        return ""

    rows = [[str(c).strip() for c in row] for row in table if any(str(c).strip() for c in row)]
    if not rows:
        return ""

    lines = [" | ".join(row) for row in rows]
    if len(lines) > 1:
        lines.insert(1, " | ".join(["---"] * len(rows[0])))

    return "\n".join(f"| {line} |" for line in lines)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(row: dict) -> str:
    """
    Combine pre_text + linearised table + post_text into a single string.
    This is the ~700-word context window that FinQA provides per example.
    """
    parts: list[str] = []

    pre = row.get("pre_text") or []
    parts.extend(p.strip() for p in pre if p.strip())

    table = row.get("table") or []
    table_str = _linearise_finqa_table(table)
    if table_str:
        parts.append(table_str)

    post = row.get("post_text") or []
    parts.extend(p.strip() for p in post if p.strip())

    return "\n\n".join(parts)


def _parse_filename(filename: str) -> tuple[str, str]:
    """
    FinQA filename format: 'AAPL/2016/table_0'
    Returns (company, year).
    """
    parts = filename.split("/")
    company = parts[0] if parts else ""
    year = parts[1] if len(parts) > 1 else ""
    return company, year


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_split(split: str) -> list[dict]:
    url = f"{FINQA_BASE}/{split}.json"
    print(f"Downloading {url} …")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    print(f"  {split}: {len(data)} examples")
    return data


# ---------------------------------------------------------------------------
# Build corpus
# ---------------------------------------------------------------------------

def build_corpus(all_splits: dict[str, list[dict]]) -> list[str]:
    """
    Build chunks.json entries from ALL split contexts (one chunk per example).
    Each chunk gets a metadata prefix from the filename.
    Returns list of "id:text" strings.
    """
    chunks: list[str] = []
    seen_ids: set[str] = set()

    for split_name, rows in all_splits.items():
        for row in rows:
            row_id = row.get("id", "")
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)

            context = _build_context(row)
            if not context.strip():
                continue

            company, year = _parse_filename(row.get("filename", ""))
            meta = f"[COMPANY: {company} | FILING: 10-K {year} | ID: {row_id}]"
            chunk_id = len(chunks)
            chunks.append(f"{chunk_id}:{meta} {context}")

    return chunks


# ---------------------------------------------------------------------------
# Build questions.json
# ---------------------------------------------------------------------------

def build_questions(rows: list[dict], chunk_id_map: dict[str, int]) -> list[dict]:
    """
    Convert FinQA examples to A-RAG questions.json format.
    chunk_id_map maps row_id → chunk_id so evidence can reference the corpus.
    """
    questions: list[dict] = []
    for row in rows:
        qa = row.get("qa", {})
        question = qa.get("question", "").strip()
        answer = str(qa.get("exe_ans", qa.get("answer", ""))).strip()

        if not question or not answer:
            continue

        # Build evidence string from gold_inds
        gold_inds = qa.get("gold_inds", {})
        evidence_parts: list[str] = []
        for key, text in gold_inds.items():
            if key.startswith("text_"):
                evidence_parts.append(text)
            elif key.startswith("table_"):
                evidence_parts.append(text)
        evidence = " | ".join(evidence_parts)

        questions.append(
            {
                "id": row.get("id", ""),
                "source": "finqa",
                "question": question,
                "answer": answer,
                "question_type": "numerical",
                "evidence": evidence,
                "evidence_relations": [],
                "program": qa.get("program", ""),
                "exe_ans": qa.get("exe_ans"),
            }
        )
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare FinQA for A-RAG evaluation")
    p.add_argument("--output", default="data/finqa", help="Output directory")
    p.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "dev", "test"],
        help="Split to use for questions.json (default: test)",
    )
    p.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence embedding model (default: all-MiniLM-L6-v2)",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Download all splits ----
    all_splits: dict[str, list[dict]] = {}
    for split in SPLITS:
        all_splits[split] = _download_split(split)

    # ---- Build corpus ----
    print("\nBuilding corpus from all splits …")
    chunks = build_corpus(all_splits)
    print(f"Total chunks: {len(chunks)}")

    chunks_file = str(output_dir / "chunks.json")
    with open(chunks_file, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Wrote {chunks_file}")

    # Build chunk_id lookup: row_id → chunk_id
    chunk_id_map: dict[str, int] = {}
    for entry in chunks:
        chunk_id_str, _, text = entry.partition(":")
        # Extract row_id from metadata prefix
        import re
        m = re.search(r"ID:\s*([^\]]+)\]", text)
        if m:
            chunk_id_map[m.group(1).strip()] = int(chunk_id_str)

    # ---- Build FAISS index ----
    index_dir = str(output_dir / "index")
    print(f"\nBuilding FAISS index in {index_dir} …")
    build_faiss_index(
        chunks_file=chunks_file,
        output_dir=index_dir,
        embedding_model=args.embedding_model,
        device=args.device,
    )

    # ---- Build questions.json ----
    eval_rows = all_splits[args.eval_split]
    print(f"\nBuilding questions.json from '{args.eval_split}' split ({len(eval_rows)} rows) …")
    questions = build_questions(eval_rows, chunk_id_map)
    print(f"  Valid questions: {len(questions)}")

    questions_file = str(output_dir / "questions.json")
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Wrote {questions_file}")

    print("\nDone. Next steps:")
    print(f"  uv run python scripts/batch_runner.py --config configs/finqa.yaml --output results/finqa --workers 3")


if __name__ == "__main__":
    main()
