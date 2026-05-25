r"""Render docs/SUBMISSION.md to docs/SUBMISSION.pdf.

Uses the `markdown-pdf` package (pure-Python, no external binaries needed
on Windows). Mermaid blocks are rendered to PNG via mermaid.ink and
embedded as images. Rendered PNGs are cached under docs/assets/ and
committed, so a working network call is only needed the first time (or
when a diagram changes).

Run:
    .\.venv\Scripts\python.exe scripts\build_pdf.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import urllib.request
import zlib
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "SUBMISSION.md"
OUT = ROOT / "docs" / "SUBMISSION.pdf"
ASSET_DIR = ROOT / "docs" / "assets"


def _mermaid_to_png(mermaid_src: str, out_path: Path) -> bool:
    """Fetch a PNG render of a mermaid diagram via mermaid.ink.

    Returns True on success, False on any failure (caller should fall
    back to a text rendering).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  cached: {out_path.name}")
        return True

    # Try several URL formats - corporate proxies + mermaid.ink behaviour
    # vary, so we attempt the most reliable encoding first.
    candidates: list[str] = []

    # 1. Plain base64 of the raw mermaid source (simplest, widely supported).
    plain = base64.b64encode(mermaid_src.encode("utf-8")).decode("ascii")
    candidates.append(
        f"https://mermaid.ink/img/{plain}?type=png&theme=default&bgColor=white"
    )

    # 2. pako-deflated JSON envelope (matches mermaid-live-editor URLs).
    try:
        payload = json.dumps(
            {"code": mermaid_src, "mermaid": {"theme": "default"}},
            separators=(",", ":"),
        ).encode("utf-8")
        # raw DEFLATE (strip 2-byte zlib header + 4-byte adler32 checksum)
        deflated = zlib.compress(payload, 9)[2:-4]
        pako = base64.urlsafe_b64encode(deflated).decode("ascii").rstrip("=")
        candidates.append(
            f"https://mermaid.ink/img/pako:{pako}?type=png&bgColor=white"
        )
    except Exception:
        pass

    for url in candidates:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (PCBuilderAgent build_pdf.py)"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if not data or len(data) < 200:
                print(f"  mermaid.ink returned empty/tiny payload ({len(data)} bytes)")
                continue
            out_path.write_bytes(data)
            print(f"  rendered: {out_path.name} ({len(data)//1024} KB)")
            return True
        except Exception as e:
            print(f"  mermaid.ink attempt failed ({type(e).__name__}): {e}")
            continue
    return False


def _render_mermaid_blocks(md_text: str) -> str:
    """Replace each ```mermaid block with an image reference."""
    pattern = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)

    def replace(match: "re.Match[str]") -> str:
        src = match.group(1).strip()
        h = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
        png_path = ASSET_DIR / f"mermaid_{h}.png"
        rel_path = os.path.relpath(png_path, SRC.parent).replace(os.sep, "/")
        print(f"Mermaid block sha1={h}:")
        ok = _mermaid_to_png(src, png_path)
        if ok:
            return f"![Architecture diagram]({rel_path})"
        # Network call failed and no cached image - keep a clearly-labeled
        # code fence so reviewers at least see the source.
        return "```text\n" + src + "\n```"

    return pattern.sub(replace, md_text)


def main() -> int:
    if not SRC.exists():
        print(f"Missing source: {SRC}", file=sys.stderr)
        return 1

    md = SRC.read_text(encoding="utf-8")

    # markdown-pdf doesn't natively render mermaid; render each block to a
    # cached PNG via mermaid.ink and replace the code fence with an image
    # link. (The original .md keeps the ```mermaid fence so GitHub still
    # renders it natively.)
    md_for_pdf = _render_mermaid_blocks(md)

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
    img { max-width: 100%; height: auto; display: block; margin: 12pt auto; }
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
