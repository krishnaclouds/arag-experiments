#!/usr/bin/env python3
"""
Prepare DocFinQA for A-RAG evaluation.

DocFinQA uses FinQA questions but grounds them in FULL SEC 10-K filings
(~150 pages / 123K words) rather than the pre-extracted FinQA contexts.
This is the most demanding financial RAG benchmark: naive single-shot
retrieval fails because the relevant evidence is buried in a dense filing.

Pipeline:
  1. Download FinQA test split (questions source)
  2. For each unique company/year, download the 10-K from SEC EDGAR
  3. Parse PDF with pdfplumber
  4. Chunk and build FAISS index
  5. Write questions.json

The SEC EDGAR full-text search API is used to locate the correct filing
accession number from the FinQA filename (e.g. "AAPL/2016/table_0").

Usage:
    uv run python scripts/prepare_docfinqa.py
    uv run python scripts/prepare_docfinqa.py --limit 20   # first 20 filings (dev run)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.indexing.build_index import build_faiss_index
from src.arag.indexing.pdf_parser import parse_pdf
from src.arag.indexing.chunker import build_chunks_json

FINQA_TEST_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
EDGAR_SEARCH   = "https://efts.sec.gov/LATEST/search-index?q={ticker}&dateRange=custom&startdt={year}-01-01&enddt={year}-12-31&forms=10-K"
EDGAR_BASE     = "https://www.sec.gov"
HEADERS        = {
    "User-Agent": "ARagPoc research@example.com",
    "Accept-Encoding": "gzip, deflate",
}


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------

def _get_edgar_10k_url(ticker: str, year: str) -> str | None:
    """
    Use SEC EDGAR full-text search to find a 10-K filing URL for (ticker, year).
    Returns the URL of the primary document, or None if not found.
    """
    try:
        # Try EDGAR company search
        cik_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?company=&CIK={ticker}&type=10-K&dateb=&owner=include"
            f"&count=10&search_text=&action=getcompany&output=atom"
        )
        req = urllib.request.Request(cik_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")

        # Extract accession numbers from EDGAR ATOM feed
        import re
        accessions = re.findall(r"/Archives/edgar/data/\d+/(\d{18})/", content)

        for acc_raw in accessions[:5]:
            acc = f"{acc_raw[:10]}-{acc_raw[10:12]}-{acc_raw[12:]}"
            filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=10&search_text="

            # Try the filing index to find the primary document
            idx_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{acc_raw[:10]}/{acc_raw}/{acc}-index.htm"
            )
            # This is a simplified heuristic; in practice CIK ≠ acc_raw[:10]
            # The sec-edgar-downloader library handles this properly
            time.sleep(0.2)

        return None  # Simplified — use sec-edgar-downloader instead
    except Exception:
        return None


def _download_via_sec_downloader(ticker: str, year: int, dest_dir: Path) -> Path | None:
    """
    Use the sec-edgar-downloader library to download a 10-K for (ticker, year).
    Returns path to the downloaded text file, or None on failure.
    """
    try:
        from sec_edgar_downloader import Downloader

        dl = Downloader("ARagPoc", "research@example.com", str(dest_dir))
        dl.get(
            "10-K",
            ticker,
            after=f"{year}-01-01",
            before=f"{year}-12-31",
            limit=1,
            download_details=False,
        )

        # sec-edgar-downloader saves to: dest_dir/sec-edgar-filings/TICKER/10-K/*/full-submission.txt
        pattern = dest_dir / "sec-edgar-filings" / ticker / "10-K" / "*" / "full-submission.txt"
        from glob import glob
        matches = glob(str(pattern))
        return Path(matches[0]) if matches else None

    except Exception as e:
        print(f"    sec-edgar-downloader error: {e}")
        return None


# ---------------------------------------------------------------------------
# Context builder (FinQA pre-extracted fallback)
# ---------------------------------------------------------------------------

def _linearise_finqa_table(table: list[list]) -> str:
    if not table:
        return ""
    rows = [[str(c).strip() for c in row] for row in table if any(str(c).strip() for c in row)]
    if not rows:
        return ""
    lines = [" | ".join(row) for row in rows]
    if len(lines) > 1:
        lines.insert(1, " | ".join(["---"] * len(rows[0])))
    return "\n".join(f"| {line} |" for line in lines)


def _finqa_context(row: dict) -> str:
    parts: list[str] = []
    for p in row.get("pre_text") or []:
        if p.strip():
            parts.append(p.strip())
    tbl = _linearise_finqa_table(row.get("table") or [])
    if tbl:
        parts.append(tbl)
    for p in row.get("post_text") or []:
        if p.strip():
            parts.append(p.strip())
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare DocFinQA (full SEC filings) for A-RAG")
    p.add_argument("--output", default="data/docfinqa")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N unique filings (default: all). Use 10-20 for a dev run.",
    )
    p.add_argument(
        "--fallback-to-context",
        action="store_true",
        default=True,
        help="If SEC filing download fails, fall back to FinQA pre-extracted context (default: True)",
    )
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    output_dir = Path(args.output)
    filings_dir = output_dir / "filings"
    output_dir.mkdir(parents=True, exist_ok=True)
    filings_dir.mkdir(parents=True, exist_ok=True)

    # ---- Download FinQA test split ----
    print(f"Downloading FinQA test split from GitHub …")
    with urllib.request.urlopen(FINQA_TEST_URL, timeout=30) as r:
        test_rows: list[dict] = json.loads(r.read())
    print(f"  {len(test_rows)} test examples")

    # ---- Group by filing (ticker/year) ----
    filing_groups: dict[str, list[dict]] = {}
    for row in test_rows:
        fname = row.get("filename", "")
        parts = fname.split("/")
        key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else fname
        filing_groups.setdefault(key, []).append(row)

    filing_keys = list(filing_groups.keys())
    if args.limit:
        filing_keys = filing_keys[: args.limit]
    print(f"Unique filings to process: {len(filing_keys)}")

    # ---- Download / parse filings ----
    all_documents: list[dict] = []
    from_edgar = 0
    from_context = 0

    for filing_key in filing_keys:
        ticker, year = filing_key.split("/")
        group = filing_groups[filing_key]
        print(f"\n[{filing_key}]  ({len(group)} questions)")

        # Attempt EDGAR download
        filing_path = _download_via_sec_downloader(ticker, int(year), filings_dir)

        if filing_path and filing_path.exists():
            print(f"  Downloaded: {filing_path.name} ({filing_path.stat().st_size // 1024} KB)")
            # full-submission.txt is a large text file, not PDF — parse as plain text
            text = filing_path.read_text(errors="replace")
            # Trim SEC header boilerplate (first 2000 chars is metadata)
            text = text[2000:] if len(text) > 2000 else text
            meta = f"[COMPANY: {ticker} | FILING: 10-K {year}]"
            all_documents.append({"text": text, "metadata": meta})
            from_edgar += 1
        elif args.fallback_to_context:
            print("  EDGAR download failed → using FinQA pre-extracted context")
            for row in group:
                ctx = _finqa_context(row)
                if ctx:
                    meta = f"[COMPANY: {ticker} | FILING: 10-K {year} | ID: {row.get('id','')}]"
                    all_documents.append({"text": ctx, "metadata": meta})
            from_context += 1

        time.sleep(0.15)  # EDGAR rate limit: ~10 req/s

    print(f"\nDocuments: {from_edgar} from EDGAR, {from_context} filings via context fallback")
    print(f"Total text sections: {len(all_documents)}")

    if not all_documents:
        print("No documents extracted — aborting.")
        sys.exit(1)

    # ---- Build chunks.json ----
    chunks_file = str(output_dir / "chunks.json")
    print(f"\nChunking …")
    build_chunks_json(all_documents, chunks_file, max_tokens=1000)

    # ---- Build FAISS index ----
    index_dir = str(output_dir / "index")
    print(f"\nBuilding FAISS index …")
    build_faiss_index(
        chunks_file=chunks_file,
        output_dir=index_dir,
        embedding_model=args.embedding_model,
        device=args.device,
    )

    # ---- Build questions.json (only for processed filings) ----
    print("\nBuilding questions.json …")
    processed_keys = set(filing_keys)
    questions: list[dict] = []

    for row in test_rows:
        fname = row.get("filename", "")
        parts = fname.split("/")
        key = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else fname
        if key not in processed_keys:
            continue

        qa = row.get("qa", {})
        question = (qa.get("question") or "").strip()
        answer = str(qa.get("exe_ans", qa.get("answer", ""))).strip()
        if not question or not answer:
            continue

        gold_inds = qa.get("gold_inds", {})
        evidence = " | ".join(v for v in gold_inds.values() if v)

        questions.append(
            {
                "id": row.get("id", ""),
                "source": "docfinqa",
                "question": question,
                "answer": answer,
                "question_type": "numerical",
                "evidence": evidence,
                "evidence_relations": [],
                "program": qa.get("program", ""),
                "exe_ans": qa.get("exe_ans"),
                "filing": key,
            }
        )

    questions_file = str(output_dir / "questions.json")
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Wrote {len(questions)} questions → {questions_file}")

    print("\nDone. Next steps:")
    print(f"  uv run python scripts/batch_runner.py --config configs/docfinqa.yaml --output results/docfinqa --workers 3")


if __name__ == "__main__":
    main()
