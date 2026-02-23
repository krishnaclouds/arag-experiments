#!/usr/bin/env python3
"""
Prepare FinanceBench for A-RAG evaluation.

Downloads the 150 open-source Q&A pairs from HuggingFace, fetches each
unique PDF (84 documents) from its doc_link URL, parses with pdfplumber,
chunks, and builds a FAISS index.

If a PDF fails to download (redirect, auth wall, rate-limit), the script
falls back to using the evidence_text field from the dataset so evaluation
can still proceed for those questions.

Usage:
    uv run python scripts/prepare_financebench.py
    uv run python scripts/prepare_financebench.py --output data/financebench --skip-download
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.indexing.build_index import build_faiss_index
from src.arag.indexing.chunker import build_chunks_json
from src.arag.indexing.pdf_parser import parse_pdf


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def _download_pdf(url: str, dest: Path, retries: int = 3, delay: float = 1.0) -> bool:
    """Download a PDF to *dest*. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 10_000:
        return True  # already downloaded

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
            # Verify it looks like a PDF
            if not content.startswith(b"%PDF"):
                print(f"    ⚠ Not a PDF (got {content[:20]!r}) — skipping")
                return False
            dest.write_bytes(content)
            return True
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code} on attempt {attempt + 1}/{retries}")
        except Exception as e:
            print(f"    Error on attempt {attempt + 1}/{retries}: {e}")
        time.sleep(delay * (attempt + 1))
    return False


# ---------------------------------------------------------------------------
# Fallback: build doc from evidence_text when PDF unavailable
# ---------------------------------------------------------------------------

def _evidence_fallback(group: list[dict]) -> list[dict]:
    """
    Build document dicts from the evidence_text fields when PDF download fails.
    Groups multiple evidence passages under the same metadata prefix.
    """
    doc_name = group[0]["doc_name"]
    company  = group[0]["company"]
    doc_type = group[0]["doc_type"].upper()
    period   = group[0]["doc_period"]
    meta = f"[COMPANY: {company} | FILING: {doc_type} {period} | DOC: {doc_name}]"

    parts: list[str] = []
    for row in group:
        for ev in (row.get("evidence") or []):
            text = (ev.get("evidence_text") or "").strip()
            if text and text not in parts:
                parts.append(text)

    if not parts:
        return []
    return [{"text": "\n\n".join(parts), "metadata": meta}]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare FinanceBench for A-RAG")
    p.add_argument("--output", default="data/financebench")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PDF download (use evidence_text fallback for all docs)",
    )
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between PDF downloads")
    args = p.parse_args()

    output_dir = Path(args.output)
    pdf_dir    = output_dir / "pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    print("Loading FinanceBench from HuggingFace …")
    from datasets import load_dataset
    ds = load_dataset("PatronusAI/financebench", split="train")
    rows = list(ds)
    print(f"Loaded {len(rows)} Q&A pairs")

    # ---- Group by unique document ----
    docs: dict[str, list[dict]] = {}
    for row in rows:
        key = row["doc_name"]
        docs.setdefault(key, []).append(row)
    print(f"Unique documents: {len(docs)}")

    # ---- Download PDFs & parse ----
    all_documents: list[dict] = []
    pdf_success = 0
    pdf_fallback = 0

    for doc_name, group in docs.items():
        first = group[0]
        url   = first["doc_link"] or ""
        company  = first["company"]
        doc_type = first["doc_type"].upper()
        period   = str(first["doc_period"])

        print(f"\n[{doc_name}]")

        if args.skip_download or not url:
            print("  → using evidence_text fallback")
            all_documents.extend(_evidence_fallback(group))
            pdf_fallback += 1
            continue

        pdf_path = pdf_dir / f"{doc_name}.pdf"
        print(f"  Downloading {url[:80]} …")
        success = _download_pdf(url, pdf_path)

        if success:
            print(f"  Parsing PDF ({pdf_path.stat().st_size // 1024} KB) …")
            try:
                doc_list = parse_pdf(
                    str(pdf_path),
                    company=company,
                    filing_type=f"{doc_type} {period}",
                    fiscal_year=period,
                )
                all_documents.extend(doc_list)
                print(f"  → {len(doc_list)} sections extracted")
                pdf_success += 1
            except Exception as e:
                print(f"  PDF parse error: {e} — falling back to evidence_text")
                all_documents.extend(_evidence_fallback(group))
                pdf_fallback += 1
        else:
            print("  → download failed, using evidence_text fallback")
            all_documents.extend(_evidence_fallback(group))
            pdf_fallback += 1

        time.sleep(args.delay)

    print(f"\nDocument summary: {pdf_success} from PDF, {pdf_fallback} from evidence fallback")
    print(f"Total text sections: {len(all_documents)}")

    if not all_documents:
        print("No documents extracted — aborting.")
        sys.exit(1)

    # ---- Build chunks.json ----
    chunks_file = str(output_dir / "chunks.json")
    print(f"\nChunking {len(all_documents)} sections …")
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

    # ---- Build questions.json ----
    print("\nBuilding questions.json …")
    questions: list[dict] = []
    for row in rows:
        question = (row.get("question") or "").strip()
        answer   = (row.get("answer") or "").strip()
        if not question or not answer:
            continue

        evidence_list = row.get("evidence") or []
        evidence = " | ".join(
            (e.get("evidence_text") or "").strip()
            for e in evidence_list
            if e.get("evidence_text")
        )

        questions.append(
            {
                "id": row["financebench_id"],
                "source": "financebench",
                "question": question,
                "answer": answer,
                "question_type": row.get("question_type", ""),
                "evidence": evidence,
                "evidence_relations": [],
                "doc_name": row["doc_name"],
                "company": row["company"],
                "doc_type": row["doc_type"],
                "doc_period": str(row["doc_period"]),
                "gics_sector": row.get("gics_sector", ""),
            }
        )

    questions_file = str(output_dir / "questions.json")
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Wrote {len(questions)} questions → {questions_file}")

    print("\nDone. Next steps:")
    print(f"  uv run python scripts/batch_runner.py --config configs/financebench.yaml --output results/financebench --workers 3")


if __name__ == "__main__":
    main()
