# Architecture (technical detail)

See [`agent_run_report.md`](agent_run_report.md) for the higher-level overview. This document drills into module boundaries and data flow.

## Module dependency graph

```mermaid
flowchart TB
    config[src/config.py] --> logging[src/logging_setup.py]
    config --> loader[src/data/loader.py]
    socket_map[src/compatibility/socket_map.py] --> loader
    loader --> schemas[src/data/schemas.py]
    loader --> search[src/tools/search.py]

    socket_map --> engine[src/compatibility/engine.py]
    memory_rules[src/compatibility/memory_rules.py] --> engine
    case_rules[src/compatibility/case_rules.py] --> engine
    power_rules[src/compatibility/power_rules.py] --> engine

    search --> registry[src/tools/registry.py]
    details[src/tools/details.py] --> registry
    compat_tool[src/tools/compatibility_tool.py] --> registry
    pricing[src/tools/pricing.py] --> registry

    config --> client[src/llm/client.py]
    providers[src/llm/providers.py] --> client

    client --> nodes[src/agent/nodes.py]
    registry --> nodes
    engine --> nodes
    prompts[src/agent/prompts.py] --> nodes
    state[src/agent/state.py] --> nodes
    guards[src/agent/guards.py] --> nodes

    nodes --> graph[src/agent/graph.py]
    memory_node[src/agent/memory.py] --> graph

    graph --> cli[src/ui/cli.py]
    graph --> streamlit_ui[src/ui/streamlit_app.py]
    graph --> eval_harness[evals/run_eval.py]
```

## Data flow per turn

1. User message arrives at the UI (`cli.py` or `streamlit_app.py`).
2. UI wraps it in `HumanMessage` and calls `graph.invoke(state, config={"thread_id":...})`.
3. LangGraph dispatches to the entry node:
   - first turn -> `gather`,
   - subsequent turns with an existing `build` in state -> `feedback`.
4. Each node returns a partial state dict; LangGraph merges it.
5. Conditional edges route based on `state["mode"]` (`plan`, `select`, `check`, `critique`, `respond`, ...).
6. `respond` writes an `AIMessage` to `state["messages"]` and the graph terminates for this turn.
7. UI extracts the latest AI message and renders it.
8. The full state (requirements, plan, build, issues, history) is persisted by `SqliteSaver` keyed on `thread_id`, so the next turn picks up where this one left off.

## Logging

Every node logs at least one `node.<name>.<event>` structured record (e.g. `node.select.pick category=cpu name="AMD Ryzen 7 7700X" price=242.98`). The structlog JSON sink writes one event per line to `traces/<run_id>.jsonl`.

## Tests

- `tests/test_socket_map.py` - microarchitecture -> socket inference.
- `tests/test_compatibility.py` - happy path + each error rule has a dedicated case.
- `tests/test_search.py` - filter operators against a mocked tiny catalog (no network).

Run with `pytest -q`.
