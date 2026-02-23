"""
PDF parser for financial documents (10-K, 10-Q, 8-K, earnings transcripts).

Extracts text and linearizes tables from PDF pages using pdfplumber.
Returns a list of document dicts compatible with build_chunks_json().
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pdfplumber


# ---------------------------------------------------------------------------
# Table linearisation
# ---------------------------------------------------------------------------

def _clean_cell(val: Any) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val)).strip()


def _linearise_table(table: list[list[Any]]) -> str:
    """
    Convert a pdfplumber table (list of rows) to pipe-delimited Markdown.

    The first non-empty row is treated as the header.
    Example output:
        | Year | Revenue | Operating Income |
        | 2023 | 383,285 | 114,301 |
        | 2022 | 394,328 | 119,437 |
    """
    if not table:
        return ""

    rows = [[_clean_cell(c) for c in row] for row in table if any(c for c in row)]
    if not rows:
        return ""

    lines = [" | ".join(row) for row in rows]
    # Insert markdown separator after header row
    if len(lines) > 1:
        lines.insert(1, " | ".join(["---"] * len(rows[0])))

    return "\n".join(f"| {line} |" for line in lines)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Common 10-K section headings
_SECTION_RE = re.compile(
    r"^(ITEM\s+\d+[A-Z]?\b[.\s].*|PART\s+[IVX]+\b.*)",
    re.IGNORECASE,
)


def _detect_section(text: str) -> str | None:
    """Return the section heading if the text starts a new 10-K section."""
    first_line = text.strip().split("\n")[0].strip()
    m = _SECTION_RE.match(first_line)
    return first_line[:80] if m else None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_pdf(
    pdf_path: str,
    company: str = "",
    filing_type: str = "",
    fiscal_year: str = "",
) -> list[dict]:
    """
    Parse a financial PDF into a list of document dicts for build_chunks_json().

    Each dict has:
      "text"     : str  — the page/section text (with tables linearised)
      "metadata" : str  — prefix tag, e.g. "[COMPANY: Apple | FILING: 10-K FY2023 | SECTION: Item 7]"

    Pages are grouped by detected section heading so that the metadata tag
    stays accurate across multi-page sections.
    """
    meta_base = _build_meta(company, filing_type, fiscal_year)
    documents: list[dict] = []
    current_section = "Unknown"
    current_pages: list[str] = []

    def _flush(section: str, pages: list[str]) -> None:
        if not pages:
            return
        full_text = "\n\n".join(pages).strip()
        if full_text:
            documents.append(
                {
                    "text": full_text,
                    "metadata": f"[{meta_base} | SECTION: {section}]",
                }
            )

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # --- extract text ---
            page_text = page.extract_text() or ""

            # --- extract and linearise tables ---
            tables = page.extract_tables() or []
            table_texts: list[str] = []
            for tbl in tables:
                lt = _linearise_table(tbl)
                if lt:
                    table_texts.append(lt)

            # Combine prose and tables (tables appended after prose for page)
            combined = page_text
            if table_texts:
                combined = combined + "\n\n" + "\n\n".join(table_texts)

            combined = combined.strip()
            if not combined:
                continue

            # --- detect section boundary ---
            detected = _detect_section(combined)
            if detected and detected != current_section:
                _flush(current_section, current_pages)
                current_section = detected
                current_pages = [combined]
            else:
                current_pages.append(combined)

    _flush(current_section, current_pages)
    return documents


def _build_meta(company: str, filing_type: str, fiscal_year: str) -> str:
    parts = []
    if company:
        parts.append(f"COMPANY: {company}")
    if filing_type:
        parts.append(f"FILING: {filing_type}")
    if fiscal_year:
        parts.append(f"FY: {fiscal_year}")
    return " | ".join(parts) if parts else "FILING"


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="Path to PDF file")
    p.add_argument("--company", default="")
    p.add_argument("--filing-type", default="10-K")
    p.add_argument("--fiscal-year", default="")
    args = p.parse_args()

    docs = parse_pdf(args.pdf, args.company, args.filing_type, args.fiscal_year)
    print(f"Extracted {len(docs)} sections")
    for i, d in enumerate(docs[:3]):
        print(f"\n--- Section {i + 1} ---")
        print(f"Metadata : {d['metadata']}")
        print(f"Text     : {d['text'][:300]}…")
