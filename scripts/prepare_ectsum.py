#!/usr/bin/env python3
"""
Prepare ECTSUM for A-RAG summarization evaluation.

Downloads 495 earnings call transcripts + expert summaries from the ECTSum
dataset, chunks each transcript, builds a FAISS sentence-level index, derives
gold_chunk_map.json for retrieval P/R evaluation, and writes questions.json.

Dataset: https://github.com/rajdeep345/ECTSum (EMNLP 2022)
HuggingFace mirror: mrSoul7766/ECTSum (primary), ChanceFocus/flare-ectsum,
                    FinanceMTEB/ECTsum

Key sentences for gold_chunk_map are derived from the expert summary lines
(each newline-separated line is one key factual sentence in the ECTSum format).

Outputs (written to --output directory):
  chunks.json          — chunked transcripts in A-RAG format
  index/               — FAISS sentence index + sentence_map.json
  questions.json       — summarization tasks in A-RAG format
  gold_chunk_map.json  — {question_id: [chunk_id, ...]} for retrieval P/R

Usage:
    uv run python scripts/prepare_ectsum.py              # full run (495 items)
    uv run python scripts/prepare_ectsum.py --limit 20   # dev run
    uv run python scripts/prepare_ectsum.py --data-dir /path/to/ECTSum/data
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.indexing.build_index import build_faiss_index
from src.arag.indexing.chunker import chunk_text

# ---------------------------------------------------------------------------
# Data loading — HuggingFace (primary) or local directory (fallback)
# ---------------------------------------------------------------------------

# Known column aliases across ECTSum dataset versions
_TRANSCRIPT_COLS = ["text", "transcript", "original", "transcript_text", "document"]
_SUMMARY_COLS    = ["summary", "reference_summary", "label", "target", "answer"]
_KEYPOINTS_COLS  = ["keypoints", "key_sentences", "kfs", "highlights", "key_points"]
_ID_COLS         = ["id", "doc_id", "filename", "file_id"]


def _find_col(row: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in row:
            return c
    return None


def _parse_keypoints(raw) -> list[str]:
    """
    Normalise the key-sentences field into a plain list of strings.
    Handles: list, newline-delimited string, JSON-encoded list.
    """
    if isinstance(raw, list):
        return [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    if isinstance(raw, str):
        # Try JSON list first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [s.strip() for s in parsed if s.strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fall back to newline / bullet splitting
        lines = re.split(r"\n+|•|\*\s+", raw)
        return [l.strip().lstrip("-•* ") for l in lines if l.strip()]
    return []


def _load_from_huggingface(limit: int | None) -> list[dict]:
    """Load via the `datasets` library from HuggingFace."""
    from datasets import load_dataset, concatenate_datasets, DatasetDict

    # Candidate datasets in priority order.
    # mrSoul7766/ECTSum test split = the 495 standard benchmark items.
    candidates = [
        ("mrSoul7766/ECTSum", "test"),          # 495 test items — standard benchmark
        ("mrSoul7766/ECTSum", "train"),          # 1681 train items — if test fails
        ("ChanceFocus/flare-ectsum", "test"),
        ("FinanceMTEB/ECTsum", "test"),
    ]
    ds = None
    for repo_id, split in candidates:
        try:
            print(f"  Trying HuggingFace dataset '{repo_id}' (split='{split}') …")
            ds = load_dataset(repo_id, split=split)
            print(f"  Loaded {len(ds)} rows from '{repo_id}/{split}'")
            break
        except Exception as e:
            print(f"  Not found: {e}")

    if ds is None:
        raise RuntimeError(
            "Could not load ECTSum from HuggingFace. "
            "Provide a local data directory with --data-dir, or check "
            "https://github.com/rajdeep345/ECTSum for the latest download instructions."
        )

    # Flatten DatasetDict if we somehow got one
    if isinstance(ds, DatasetDict):
        ds = concatenate_datasets(list(ds.values()))

    rows = list(ds)
    if limit:
        rows = rows[:limit]
    return rows


def _load_from_local_dir(data_dir: Path, limit: int | None) -> list[dict]:
    """Load from a local directory containing CSV/JSON files from the ECTSum repo."""
    import csv

    rows: list[dict] = []
    # Accept .csv, .tsv, .jsonl, .json
    files = sorted(
        list(data_dir.glob("*.csv"))
        + list(data_dir.glob("*.tsv"))
        + list(data_dir.glob("*.jsonl"))
        + list(data_dir.glob("*.json"))
    )
    if not files:
        raise FileNotFoundError(f"No CSV/JSON files found in {data_dir}")

    print(f"  Found {len(files)} data file(s): {[f.name for f in files]}")

    for fpath in files:
        if fpath.suffix in {".csv", ".tsv"}:
            delim = "\t" if fpath.suffix == ".tsv" else ","
            with open(fpath, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delim)
                rows.extend(list(reader))
        elif fpath.suffix == ".jsonl":
            with open(fpath, encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        elif fpath.suffix == ".json":
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    rows.extend(data)
                elif isinstance(data, dict):
                    rows.extend(data.values())

    if limit:
        rows = rows[:limit]
    return rows


# ---------------------------------------------------------------------------
# Gold chunk map builder
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for robust substring matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def build_gold_chunk_map(
    question_id: str,
    key_sentences: list[str],
    doc_chunks: list[tuple[int, str]],  # [(chunk_id, chunk_text), ...]
    min_match_chars: int = 40,
) -> list[int]:
    """
    For a single question, return the IDs of chunks that contain at least
    one key sentence (or a substantial substring of it, to handle chunker
    splits across sentence boundaries).
    """
    gold_ids: set[int] = set()

    for sentence in key_sentences:
        norm_sent = _normalise(sentence)
        if len(norm_sent) < min_match_chars:
            continue  # too short to be a reliable anchor

        # Try full match first; fall back to first 40-char prefix
        anchors = [norm_sent, norm_sent[:min_match_chars]]

        for chunk_id, chunk_text_str in doc_chunks:
            norm_chunk = _normalise(chunk_text_str)
            if any(anchor in norm_chunk for anchor in anchors):
                gold_ids.add(chunk_id)
                break  # found the chunk for this sentence; move to next sentence

    return sorted(gold_ids)


# ---------------------------------------------------------------------------
# chunks.json builder (custom — tracks per-doc chunk IDs)
# ---------------------------------------------------------------------------

def build_chunks_for_docs(
    docs: list[dict],   # [{"metadata": str, "text": str}, ...]
    output_file: str,
    max_tokens: int = 1000,
) -> tuple[list[str], list[list[tuple[int, str]]]]:
    """
    Build chunks.json and return per-document chunk lists for gold_chunk_map.

    Returns:
        all_chunks  : list of "id:text" strings (written to output_file)
        doc_chunks  : list where doc_chunks[i] = [(chunk_id, text), ...] for docs[i]
    """
    all_chunks: list[str] = []
    doc_chunk_lists: list[list[tuple[int, str]]] = []
    chunk_id = 0

    for doc in docs:
        text = doc["text"]
        prefix = doc.get("metadata", "")
        this_doc: list[tuple[int, str]] = []

        for body in chunk_text(text, max_tokens=max_tokens):
            full = f"{prefix} {body}".strip() if prefix else body
            all_chunks.append(f"{chunk_id}:{full}")
            this_doc.append((chunk_id, full))
            chunk_id += 1

        doc_chunk_lists.append(this_doc)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Wrote {len(all_chunks)} chunks → {output_file}")
    return all_chunks, doc_chunk_lists


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare ECTSUM for A-RAG summarization eval")
    p.add_argument("--output", default="data/ectsum", help="Output directory")
    p.add_argument("--data-dir", default=None, help="Local ECTSum data directory (skips HuggingFace download)")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N transcripts (dev mode)")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-tokens", type=int, default=1000, help="Chunk size in tokens (default 1000)")
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load raw data ----
    print("Loading ECTSum data …")
    if args.data_dir:
        rows = _load_from_local_dir(Path(args.data_dir), args.limit)
    else:
        rows = _load_from_huggingface(args.limit)

    print(f"Loaded {len(rows)} samples")
    if not rows:
        print("No data found — aborting.")
        sys.exit(1)

    # ---- Detect column names from first row ----
    first = rows[0]
    transcript_col = _find_col(first, _TRANSCRIPT_COLS)
    summary_col    = _find_col(first, _SUMMARY_COLS)
    keypoints_col  = _find_col(first, _KEYPOINTS_COLS)
    id_col         = _find_col(first, _ID_COLS)

    if not transcript_col:
        print(f"ERROR: Could not find transcript column. Available columns: {list(first.keys())}")
        sys.exit(1)
    if not summary_col:
        print(f"ERROR: Could not find summary column. Available columns: {list(first.keys())}")
        sys.exit(1)

    print(f"Column mapping: transcript='{transcript_col}', summary='{summary_col}', "
          f"keypoints='{keypoints_col or 'NOT FOUND'}', id='{id_col or 'auto'}'")

    # ---- Build document list ----
    docs: list[dict] = []       # for chunking
    samples: list[dict] = []    # cleaned sample dicts

    for i, row in enumerate(rows):
        transcript = str(row.get(transcript_col, "")).strip()
        summary    = str(row.get(summary_col, "")).strip()

        if not transcript or not summary:
            continue  # skip empty rows

        # Use keypoints column if present; otherwise derive key sentences from
        # summary lines (each line in the ECTSum summary is a key factual sentence).
        if keypoints_col and row.get(keypoints_col):
            key_sentences = _parse_keypoints(row[keypoints_col])
        else:
            # Split summary by newlines — each line is one KFS in ECTSum format
            key_sentences = [
                s.strip().lstrip("-•* ").strip()
                for s in summary.splitlines()
                if s.strip()
            ]

        # Derive a stable ID
        if id_col and row.get(id_col):
            raw_id = str(row[id_col]).strip().replace(" ", "_")
            question_id = f"ectsum_{raw_id}"
        else:
            question_id = f"ectsum_{i:04d}"

        # Extract company / period from the ID if it follows TICKER_Qn_YYYY format.
        # If the ID is just a numeric index (no ticker info), leave company/period empty
        # so the question text degrades gracefully.
        raw_suffix = question_id.replace("ectsum_", "")
        parts = raw_suffix.split("_")
        # Only treat parts as company/period if the first token looks like a ticker
        # (alphabetic, 1-5 chars) rather than a plain zero-padded number.
        if len(parts) >= 2 and parts[0].isalpha() and len(parts[0]) <= 5:
            company = parts[0].upper()
            period  = "_".join(parts[1:])
        else:
            company = ""
            period  = ""

        metadata = (
            f"[COMPANY: {company} | TYPE: EARNINGS_CALL | "
            f"PERIOD: {period} | SECTION: transcript]"
        )

        docs.append({"text": transcript, "metadata": metadata})
        samples.append({
            "id": question_id,
            "company": company,
            "period": period,
            "summary": summary,
            "key_sentences": key_sentences,
        })

    print(f"Valid samples after filtering: {len(samples)}")

    # ---- Build chunks.json (tracking per-doc chunk IDs) ----
    chunks_file = str(output_dir / "chunks.json")
    print(f"\nChunking {len(docs)} transcripts (max_tokens={args.max_tokens}) …")
    _, doc_chunk_lists = build_chunks_for_docs(docs, chunks_file, max_tokens=args.max_tokens)

    # ---- Build gold_chunk_map.json ----
    print("\nBuilding gold_chunk_map.json …")
    gold_chunk_map: dict[str, list[int]] = {}
    no_kfs_count = 0
    no_match_count = 0

    for sample, doc_chunks in zip(samples, doc_chunk_lists):
        qid = sample["id"]
        kfs = sample["key_sentences"]

        if not kfs:
            no_kfs_count += 1
            gold_chunk_map[qid] = []
            continue

        gold_ids = build_gold_chunk_map(qid, kfs, doc_chunks)
        gold_chunk_map[qid] = gold_ids
        if not gold_ids:
            no_match_count += 1

    gold_map_file = output_dir / "gold_chunk_map.json"
    with open(gold_map_file, "w") as f:
        json.dump(gold_chunk_map, f, indent=2)

    covered = sum(1 for v in gold_chunk_map.values() if v)
    print(f"gold_chunk_map: {covered}/{len(samples)} questions have ≥1 gold chunk")
    if no_kfs_count:
        print(f"  {no_kfs_count} samples had no key sentences (retrieval P/R will be 0)")
    if no_match_count:
        print(f"  {no_match_count} samples had key sentences but no chunk matches "
              f"(check chunking alignment)")
    print(f"Wrote {gold_map_file}")

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
    questions: list[dict] = []

    for sample in samples:
        company = sample["company"]
        period  = sample["period"].replace("_", " ") if sample["period"] else ""
        if company and period:
            q_text = f"Summarize the key highlights from the {company} {period} earnings call."
        elif company:
            q_text = f"Summarize the key highlights from this {company} earnings call."
        else:
            q_text = "Summarize the key highlights from this earnings call."

        questions.append({
            "id": sample["id"],
            "source": "ectsum",
            "question": q_text,
            "answer": sample["summary"],
            "question_type": "summarization",
            "summarization_style": "earnings_call",
            "evidence": " | ".join(sample["key_sentences"]),
            "evidence_relations": [],
        })

    questions_file = str(output_dir / "questions.json")
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Wrote {len(questions)} questions → {questions_file}")

    # ---- Summary ----
    total_chunks = sum(len(dc) for dc in doc_chunk_lists)
    avg_chunks = total_chunks / len(samples) if samples else 0
    kfs_coverage = covered / len(samples) if samples else 0

    print(f"\n{'=' * 55}")
    print(f"ECTSUM preparation complete:")
    print(f"  Transcripts processed : {len(samples)}")
    print(f"  Total chunks          : {total_chunks} (avg {avg_chunks:.1f} per transcript)")
    print(f"  Gold chunk coverage   : {kfs_coverage:.1%} of questions")
    print(f"  Output directory      : {output_dir}/")
    print(f"{'=' * 55}")
    print("\nNext steps:")
    print(f"  # Dev run (20 questions):")
    print(f"  uv run python scripts/batch_runner.py \\")
    print(f"    --config configs/ectsum.yaml --output results/ectsum --workers 3 --limit 20")
    print(f"\n  # Evaluate:")
    print(f"  uv run python scripts/eval.py \\")
    print(f"    --predictions results/ectsum/predictions.jsonl \\")
    print(f"    --task-type summarization \\")
    print(f"    --gold-chunk-map data/ectsum/gold_chunk_map.json")


if __name__ == "__main__":
    main()
