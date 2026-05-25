r"""Render docs/SUBMISSION.md to docs/SUBMISSION.pdf.

Uses the `markdown-pdf` package (pure-Python, no external binaries needed
on Windows).

Run:
    .\.venv\Scripts\python.exe scripts\build_pdf.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "SUBMISSION.md"
OUT = ROOT / "docs" / "SUBMISSION.pdf"


def main() -> int:
    if not SRC.exists():
        print(f"Missing source: {SRC}", file=sys.stderr)
        return 1

    md = SRC.read_text(encoding="utf-8")

    # markdown-pdf doesn't natively render mermaid; replace mermaid fences with
    # plain code fences so the diagram source still appears in the PDF rather
    # than disappearing. The mermaid block remains valid in the .md file for
    # GitHub's renderer.
    md_for_pdf = md.replace("```mermaid", "```text")

    # markdown-pdf builds its own TOC; remove our manual [text](#anchor) links
    # because they would otherwise raise "No destination with id=..." (it does
    # not auto-generate the same slug ids as GitHub).
    md_for_pdf = re.sub(r"\[([^\]]+)\]\(#[^\)]+\)", r"\1", md_for_pdf)

    pdf = MarkdownPdf(toc_level=2, optimize=True)
    css = """
    body { font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.45; }
    h1 { font-size: 24pt; border-bottom: 2px solid #444; padding-bottom: 4pt; }
    h2 { font-size: 18pt; border-bottom: 1px solid #888; padding-bottom: 2pt; margin-top: 24pt; }
    h3 { font-size: 14pt; margin-top: 18pt; }
    h4 { font-size: 12pt; margin-top: 14pt; }
    code, pre { font-family: Consolas, "Courier New", monospace; font-size: 9.5pt; }
    pre { background: #f4f4f4; padding: 8pt; border-radius: 4pt; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #888; padding: 4pt 7pt; vertical-align: top; }
    th { background: #eee; }
    blockquote { color: #555; border-left: 4px solid #888; padding-left: 10pt; margin-left: 0; }
    """
    pdf.add_section(Section(md_for_pdf, toc=True), user_css=css)
    pdf.meta["title"] = "PC Builder Agent - Submission Document"
    pdf.meta["author"] = "Hrithik"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(OUT)
    size_kb = OUT.stat().st_size // 1024
    print(f"Wrote {OUT} ({size_kb} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
