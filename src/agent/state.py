"""LangGraph state definition."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """The graph state. Each node reads & updates a slice of this dict."""

    # Conversation
    messages: Annotated[List[BaseMessage], add_messages]

    # Extracted requirements (dict serialization of Requirements pydantic model)
    requirements: Optional[Dict[str, Any]]

    # Planner output
    plan: Optional[Dict[str, Any]]

    # Selected build so far (category -> component dict)
    build: Optional[Dict[str, Any]]

    # Snapshot of the previous build (set by feedback_handler on revisions).
    # Used by the responder to produce a build-vs-build comparison section.
    previous_build: Optional[Dict[str, Any]]
    previous_budget_usd: Optional[float]

    # Compatibility issues from the most recent check
    compat_issues: List[Dict[str, Any]]

    # Self-critique decision
    critique: Optional[Dict[str, Any]]

    # Last user feedback parsed into a delta
    feedback: Optional[Dict[str, Any]]

    # Counters / control
    iteration: int
    selector_attempts: int
    critique_attempts: int

    # Final assistant message content (filled by responder)
    final_response: Optional[str]

    # Mode indicator - controls graph routing
    # Values: "gather" | "plan" | "select" | "respond" | "feedback"
    mode: str
