"""Rich-based interactive CLI."""
from __future__ import annotations

import argparse
import sys
import uuid
from typing import Optional

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.agent.graph import build_graph, make_thread_config
from src.config import get_settings
from src.logging_setup import configure_logging, current_trace_path


console = Console()


def _show_banner():
    settings = get_settings()
    console.print(Panel.fit(
        "[bold cyan]PC Builder Agent[/bold cyan]\n"
        f"Provider: [green]{settings.llm_provider}[/green] / "
        f"Model: [green]{settings.ollama_model}[/green]\n"
        "Type your request (e.g. '1500 USD gaming PC'). "
        "Type [yellow]/exit[/yellow] to quit, [yellow]/new[/yellow] to start "
        "a new build, [yellow]/trace[/yellow] to print the trace path.",
        title="Welcome",
    ))


def _render_assistant(content: str):
    console.print(Panel(Markdown(content or "(empty response)"),
                        title="Assistant", border_style="cyan"))


def _print_quick_status(state: dict):
    build = state.get("build") or {}
    if not build:
        return
    table = Table(title="Current build (quick view)", show_lines=False)
    table.add_column("Component", style="bold")
    table.add_column("Part")
    table.add_column("Price", justify="right")
    total = 0.0
    for cat, comp in build.items():
        if comp:
            table.add_row(cat, str(comp.get("name", "?")),
                          f"${float(comp.get('price', 0)):.2f}")
            total += float(comp.get("price", 0) or 0)
    table.add_section()
    table.add_row("", "[bold]Total[/bold]", f"[bold]${total:.2f}[/bold]")
    console.print(table)


def run_cli(initial_message: Optional[str] = None, thread_id: Optional[str] = None) -> str:
    """Run the interactive CLI loop. Returns the final assistant message
    (useful for scripting and evaluation runs)."""
    run_id = configure_logging()
    settings = get_settings()
    _show_banner()

    graph = build_graph(with_memory=True)
    thread_id = thread_id or f"cli-{uuid.uuid4().hex[:8]}"
    config = make_thread_config(thread_id)

    console.print(f"[dim]Thread:[/dim] {thread_id}  "
                  f"[dim]Trace:[/dim] {current_trace_path()}")

    last_response = ""
    first_iter = True
    while True:
        if first_iter and initial_message:
            user_text = initial_message
            first_iter = False
            console.print(f"[bold]You:[/bold] {user_text}")
        else:
            try:
                user_text = Prompt.ask("[bold]You[/bold]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/dim]")
                return last_response

        cmd = user_text.strip().lower()
        if cmd in {"/exit", "/quit", ":q"}:
            console.print("[dim]Bye.[/dim]")
            return last_response
        if cmd == "/new":
            thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            config = make_thread_config(thread_id)
            console.print(f"[yellow]Started new thread:[/yellow] {thread_id}")
            continue
        if cmd == "/trace":
            console.print(f"[dim]Trace file:[/dim] {current_trace_path()}")
            continue

        state_input = {"messages": [HumanMessage(content=user_text)]}
        try:
            result = graph.invoke(state_input, config=config)
        except Exception as e:
            console.print(f"[red]Agent error:[/red] {e}")
            continue

        # Find the most recent AI message in the merged state
        msgs = result.get("messages") or []
        assistant_msgs = [m for m in msgs if getattr(m, "type", "") == "ai"]
        if assistant_msgs:
            last_response = assistant_msgs[-1].content
        else:
            last_response = result.get("final_response") or "(no response)"

        _print_quick_status(result)
        _render_assistant(last_response)

        if initial_message and not sys.stdin.isatty():
            # One-shot mode: return after the first response.
            return last_response


def main():
    parser = argparse.ArgumentParser(description="PC Builder Agent - CLI")
    parser.add_argument("-m", "--message", help="One-shot user message", default=None)
    parser.add_argument("-t", "--thread", help="Reuse a specific thread_id", default=None)
    args = parser.parse_args()
    run_cli(initial_message=args.message, thread_id=args.thread)


if __name__ == "__main__":
    main()
