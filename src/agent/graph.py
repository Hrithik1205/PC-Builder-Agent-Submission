"""LangGraph construction: wire nodes + edges + conditional routing.

The graph implements the reason -> plan -> act -> observe -> respond loop:

  ENTRY -> gather -> [respond | plan]
  plan -> select -> check
  check -> [select (errors) | critique (ok)]
  critique -> [select (revise) | respond (approve)]
  respond -> END
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

from src.agent.memory import get_checkpointer
from src.agent.nodes import (
    compatibility_checker,
    component_selector,
    feedback_handler,
    planner,
    requirement_gatherer,
    responder,
    self_critique,
)
from src.agent.state import AgentState
from src.config import get_settings
from src.logging_setup import get_logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conditional routers
# ---------------------------------------------------------------------------

def _route_after_gather(state: AgentState) -> str:
    return state.get("mode", "plan")


def _route_after_check(state: AgentState) -> str:
    mode = state.get("mode", "critique")
    # Loop guard: respect max_agent_steps overall
    settings = get_settings()
    if state.get("selector_attempts", 0) >= settings.max_agent_steps // 5:
        return "critique"
    return mode


def _route_after_critique(state: AgentState) -> str:
    return state.get("mode", "respond")


def _route_entry(state: AgentState) -> str:
    """Decide whether a new turn is initial-request or follow-up feedback."""
    # If we have a build already, this turn is feedback.
    if state.get("build"):
        return "feedback"
    return "gather"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(with_memory: bool = True):
    """Compile and return the agent graph.

    If `with_memory` is True, attaches a SqliteSaver checkpointer so the
    graph remembers per-thread state between turns.
    """
    g: StateGraph = StateGraph(AgentState)

    g.add_node("gather", requirement_gatherer)
    g.add_node("plan", planner)
    g.add_node("select", component_selector)
    g.add_node("check", compatibility_checker)
    g.add_node("critique", self_critique)
    g.add_node("respond", responder)
    g.add_node("feedback", feedback_handler)

    # Entry routing
    g.add_conditional_edges(
        START, _route_entry,
        {"gather": "gather", "feedback": "feedback"},
    )

    # Linear backbone + branches
    g.add_conditional_edges(
        "gather", _route_after_gather,
        {"plan": "plan", "respond": "respond"},
    )
    g.add_edge("plan", "select")
    g.add_edge("select", "check")
    g.add_conditional_edges(
        "check", _route_after_check,
        {"select": "select", "critique": "critique"},
    )
    g.add_conditional_edges(
        "critique", _route_after_critique,
        {"select": "select", "respond": "respond"},
    )
    g.add_edge("respond", END)

    # Feedback path: either re-plan or short-circuit to respond
    g.add_conditional_edges(
        "feedback", lambda s: s.get("mode", "plan"),
        {"plan": "plan", "respond": "respond"},
    )

    checkpointer = get_checkpointer() if with_memory else None
    compiled = g.compile(checkpointer=checkpointer)
    log.info("graph.compiled", with_memory=with_memory)
    return compiled


def make_thread_config(thread_id: str) -> dict:
    """Build the `configurable` dict LangGraph expects for checkpointing."""
    settings = get_settings()
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_agent_steps,
    }
