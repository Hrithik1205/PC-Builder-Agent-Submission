"""Render an agent run trace (jsonl) into a readable Markdown narrative.

Each line in the trace is a JSON object emitted by structlog. The renderer
groups events by `event` prefix (`node.gather.*`, `node.plan.*`, etc.) and
emits one section per node + a chronological event list at the bottom.

Usage:
    python -m scripts.render_trace traces/20260520T....jsonl > docs/trace_example.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _format_event_inline(ev: Dict[str, Any]) -> str:
    main = ev.get("event", "?")
    extras = {k: v for k, v in ev.items()
              if k not in {"event", "timestamp", "level", "run_id", "logger"}}
    extras_str = ", ".join(f"{k}={_short(v)}" for k, v in extras.items())
    ts = ev.get("timestamp", "")
    return f"- `{ts}` **{main}** {extras_str}"


def _short(v: Any, max_len: int = 140) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def render(path: Path) -> str:
    events = _load_jsonl(path)
    if not events:
        return f"# Trace {path.name}\n\n_(no events parsed)_"

    out: List[str] = []
    out.append(f"# Agent Trace - `{path.name}`")
    out.append("")
    out.append(f"_{len(events)} events_")
    out.append("")

    # Per-node summary
    nodes = ["gather", "plan", "select", "check", "critique", "respond", "feedback"]
    for n in nodes:
        block = [e for e in events
                 if e.get("event", "").startswith(f"node.{n}.")]
        if not block:
            continue
        out.append(f"## Node: `{n}`")
        out.append("")
        for e in block:
            out.append(_format_event_inline(e))
        out.append("")

    # LLM and tool calls
    llm_calls = [e for e in events if e.get("event") == "llm.invoke"]
    if llm_calls:
        out.append("## LLM invocations")
        out.append("")
        out.append("| # | latency_ms | input_tokens | output_tokens | tool_calls | mode |")
        out.append("|---|---|---|---|---|---|")
        for i, e in enumerate(llm_calls, 1):
            out.append(
                f"| {i} | {e.get('latency_ms','')} | {e.get('input_tokens','')} | "
                f"{e.get('output_tokens','')} | {e.get('tool_calls','')} | {e.get('mode','')} |"
            )
        out.append("")

    # Full chronological log
    out.append("## Full chronological log")
    out.append("")
    for e in events:
        out.append(_format_event_inline(e))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Render a jsonl trace to markdown")
    parser.add_argument("trace", help="Path to a traces/*.jsonl file")
    parser.add_argument("-o", "--output", help="Output markdown path (default: stdout)",
                        default=None)
    args = parser.parse_args()

    path = Path(args.trace)
    if not path.exists():
        print(f"Trace not found: {path}", file=sys.stderr)
        sys.exit(1)

    md = render(path)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Rendered to {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
