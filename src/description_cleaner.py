"""Clean raw HTML job descriptions into structured text suitable for LLM summarization.

The workflow here converts HTML content into plain text, preserves likely section
headers, strips generic boilerplate language, and writes one JSONL record per job
with a cleaned description field.
"""

from __future__ import annotations
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

INPUT_XLSX = r"D:\Assignment\RAG\Data\cleaned\description.xlsx"
OUTPUT_JSONL = r"D:\Assignment\RAG\Data\json\description_clean.jsonl"
ID_COL = "ID"
DESC_COL = "Job Description"
STRIP_BOILERPLATE = True

BOILERPLATE_PATTERNS = [
    r"qualified applicants with arrest or conviction records.*?laws\.?",
    r"it is the policy of the firm to ensure equal employment opportunity.*?law\.?",
    r"\w[\w\s]*? is an equal opportunity employer[^.]*\.?",
    r"we anticipate the application window for this opening will close on:?\s*\d{1,2}/\d{1,2}/\d{2,4}\.?",
    r"equal employment opportunity/m/f/vet/disabled\.?",
]

HEADER_TAGS = {"b", "strong"}


def _is_probable_header(text: str) -> bool:
    """Heuristic used to detect section labels such as Responsibilities or Requirements."""
    text = text.strip()
    if not text or len(text) > 60:
        return False
    if text.endswith((".", ",", ";", ":")) and not text.endswith(":"):
        return False
    word_count = len(text.split())
    return 1 <= word_count <= 8


def html_to_structured_text(raw_html: str) -> str:
    """Convert HTML into a readable text form that preserves headings and item lists.

    The function traverses the DOM tree, recognizes likely section headers,
    formats them as `## Header`, and emits list items as `- ...` so downstream
    LLM prompts retain structure.
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    lines = []
    buffer = []  

    def flush():
        text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        buffer.clear()
        if text:
            lines.append(text)

    def append_inline(text):
        text = text.strip()
        if text:
            buffer.append(text)

    def walk(node):
        if isinstance(node, NavigableString):
            append_inline(str(node))
            return

        if not isinstance(node, Tag):
            return

        name = node.name.lower()

        if name in ("script", "style"):
            return

        if name == "li":
            flush()
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(f"- {text}")
            return

        if name in HEADER_TAGS:
            text = node.get_text(" ", strip=True)
            next_sib = node.next_sibling
            continues_sentence = (
                isinstance(next_sib, NavigableString)
                and next_sib.strip()
                and next_sib.strip()[0].islower()
            )
            if _is_probable_header(text) and not continues_sentence:
                flush()
                lines.append("")
                lines.append(f"## {text.rstrip(':')}")
                lines.append("")
                return
            
            append_inline(text)
            return

        if name == "br":
            flush()
            return

        if name in ("p", "div", "ul", "ol"):
            for child in node.children:
                walk(child)
            flush()
            lines.append("")  
            return

        for child in node.children:
            walk(child)

    for child in soup.children:
        walk(child)
    flush()

    text = "\n".join(lines)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace while preserving meaningful line breaks."""
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    """Remove repeated recruitment boilerplate and legal footer language."""
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def clean_one(raw_html: str, strip_bp: bool) -> str:
    """Apply the full HTML-to-text cleaning workflow to a single job description."""
    text = html_to_structured_text(raw_html)
    text = normalize_whitespace(text)
    if strip_bp:
        text = strip_boilerplate(text)
        text = normalize_whitespace(text)
    return text


def main():
    """Run the cleaning pass over the input workbook and write the cleaned JSONL output."""
    in_path = Path(INPUT_XLSX)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    df = pd.read_excel(in_path)
    missing = {ID_COL, DESC_COL} - set(df.columns)
    if missing:
        sys.exit(
            f"Missing expected column(s) {missing} in {in_path}. "
            f"Found columns: {list(df.columns)}"
        )

    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written, n_empty = 0, 0
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            job_id = str(row[ID_COL]).strip()
            raw = row[DESC_COL]
            cleaned = clean_one(raw, strip_bp=STRIP_BOILERPLATE)
            if not cleaned:
                n_empty += 1
            record = {"ID": job_id, "job_description_clean": cleaned}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {out_path} ({n_empty} had empty/blank descriptions).")


if __name__ == "__main__":
    main()