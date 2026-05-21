"""Scenario-based evaluation harness.

Runs each scenario from `scenarios.yaml` through the agent graph and checks
the result against quantitative + qualitative expectations. Writes a
timestamped Markdown report to `evals/reports/`.

Usage:
    python -m evals.run_eval                  # run all scenarios
    python -m evals.run_eval --only gaming_1500
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml
from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph, make_thread_config
from src.compatibility.engine import check_build, has_errors
from src.data.schemas import Build
from src.logging_setup import configure_logging


ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = ROOT / "evals" / "scenarios.yaml"
REPORT_DIR = ROOT / "evals" / "reports"


def _load_scenarios() -> List[Dict[str, Any]]:
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["scenarios"]


def _run_scenario(graph, scenario: Dict[str, Any]) -> Dict[str, Any]:
    sid = scenario["id"]
    thread = f"eval-{sid}-{uuid.uuid4().hex[:6]}"
    config = make_thread_config(thread)
    out: Dict[str, Any] = {"scenario_id": sid, "thread": thread,
                           "started_at": dt.datetime.utcnow().isoformat()}

    t0 = time.time()
    # Initial turn
    initial = scenario.get("initial_message") or scenario.get("message")
    state_in = {"messages": [HumanMessage(content=initial)]}
    result = graph.invoke(state_in, config=config)
    out["initial_response"] = _extract_response(result)
    out["initial_build"] = result.get("build") or {}

    # Feedback turn (if any)
    if "feedback_message" in scenario:
        state_in = {"messages": [HumanMessage(content=scenario["feedback_message"])]}
        result = graph.invoke(state_in, config=config)
        out["feedback_response"] = _extract_response(result)
        out["feedback_build"] = result.get("build") or {}

    out["final_build"] = result.get("build") or {}
    out["final_response"] = _extract_response(result)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def _extract_response(result: Dict[str, Any]) -> str:
    msgs = result.get("messages") or []
    for m in reversed(msgs):
        if getattr(m, "type", "") == "ai":
            return m.content or ""
    return result.get("final_response") or ""


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def _check_scenario(scenario: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    expected = scenario.get("expected", {})
    build = run.get("final_build") or {}
    response = (run.get("final_response") or "").lower()
    checks: List[Dict[str, Any]] = []

    def ok(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    spb = expected.get("should_produce_build")
    if spb is True:
        ok("produced_build", bool(build), f"selected={list(build.keys())}")
    elif spb is False:
        ok("did_not_produce_build", not build, f"selected={list(build.keys())}")
    elif spb == "false_or_warn":
        # Either no build, OR response includes one of the keywords below
        ok_phrase = any(
            p in response for p in expected.get("response_must_mention_any_of", [])
        )
        ok("infeasibility_acknowledged",
           (not build) or ok_phrase,
           f"build_empty={not build}, response_keyword_present={ok_phrase}")

    if "max_total_price" in expected and build:
        try:
            total = Build(**build).total_price()
            ok("price_under_max", total <= expected["max_total_price"],
               f"total=${total} max=${expected['max_total_price']}")
            ok("price_above_min", total >= expected.get("min_total_price", 0),
               f"total=${total}")
        except Exception as e:
            ok("price_under_max", False, f"could not compute: {e}")

    for cat in expected.get("required_categories", []):
        ok(f"has_{cat}", cat in build and build.get(cat),
           f"value={build.get(cat, {}).get('name') if build.get(cat) else None}")

    if "min_memory_gb" in expected and build.get("memory"):
        gb = build["memory"].get("total_gb")
        ok("min_memory_gb",
           (gb or 0) >= expected["min_memory_gb"],
           f"got={gb} GB, min={expected['min_memory_gb']} GB")

    if "min_storage_gb" in expected and build.get("storage"):
        gb = build["storage"].get("capacity")
        ok("min_storage_gb", (gb or 0) >= expected["min_storage_gb"],
           f"got={gb} GB")

    if "min_psu_watts" in expected and build.get("power_supply"):
        w = build["power_supply"].get("wattage")
        ok("min_psu_watts", (w or 0) >= expected["min_psu_watts"],
           f"got={w}W")

    if expected.get("must_have_discrete_gpu") is True:
        ok("has_discrete_gpu", bool(build.get("video_card")),
           f"video_card={build.get('video_card', {}).get('name')}")
    elif expected.get("must_have_discrete_gpu") is False:
        # No-op: discrete GPU is allowed but not required
        pass

    if expected.get("no_compatibility_errors") is True and build:
        try:
            issues = check_build(Build(**build))
            ok("no_compatibility_errors", not has_errors(issues),
               f"errors={[i.message for i in issues if i.severity=='error']}")
        except Exception as e:
            ok("no_compatibility_errors", False, f"exception: {e}")

    if "response_must_mention_any_of" in expected and "feedback_message" in scenario:
        keywords = expected["response_must_mention_any_of"]
        present = any(k.lower() in response for k in keywords)
        ok("response_mentions_keyword", present,
           f"required_any_of={keywords}")

    passed = sum(1 for c in checks if c["passed"])
    return {"checks": checks, "passed": passed, "total": len(checks)}


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _render_report(runs: List[Dict[str, Any]], checks: List[Dict[str, Any]]) -> str:
    lines = ["# PC Builder Agent - Evaluation Report",
             f"_Generated: {dt.datetime.utcnow().isoformat()}Z_", ""]
    total_pass = sum(c["passed"] for c in checks)
    total_all = sum(c["total"] for c in checks)
    lines.append(f"**Overall: {total_pass} / {total_all} checks passed.**")
    lines.append("")
    for run, check_block in zip(runs, checks):
        sid = run["scenario_id"]
        lines.append(f"## Scenario `{sid}`")
        lines.append(f"- Elapsed: {run.get('elapsed_s')}s")
        lines.append(f"- Final price: ${Build(**(run['final_build'] or {})).total_price()}")
        lines.append(f"- Passed: **{check_block['passed']} / {check_block['total']}**")
        lines.append("")
        lines.append("### Checks")
        for c in check_block["checks"]:
            tick = "PASS" if c["passed"] else "FAIL"
            lines.append(f"- [{tick}] `{c['check']}` - {c['detail']}")
        lines.append("")
        lines.append("### Build")
        build = run.get("final_build") or {}
        if build:
            lines.append("| Component | Part | Price |")
            lines.append("|---|---|---|")
            for cat, comp in build.items():
                if comp:
                    lines.append(
                        f"| {cat} | {comp.get('name')} | "
                        f"${float(comp.get('price', 0)):.2f} |"
                    )
        else:
            lines.append("_(no build produced)_")
        lines.append("")
        lines.append("### Final response")
        lines.append("```")
        lines.append((run.get("final_response") or "")[:2000])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run agent evaluation scenarios")
    parser.add_argument("--only", help="Run a single scenario by id", default=None)
    args = parser.parse_args()

    configure_logging()
    scenarios = _load_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if s["id"] == args.only]
        if not scenarios:
            print(f"No scenario with id={args.only}")
            sys.exit(1)

    graph = build_graph(with_memory=True)
    runs, checks = [], []
    for sc in scenarios:
        print(f"==> Running {sc['id']}...", flush=True)
        run = _run_scenario(graph, sc)
        result = _check_scenario(sc, run)
        runs.append(run)
        checks.append(result)
        print(f"   {result['passed']}/{result['total']} checks passed", flush=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    report_path = REPORT_DIR / f"eval_{ts}.md"
    report_path.write_text(_render_report(runs, checks), encoding="utf-8")
    # Also dump raw json for debugging
    (REPORT_DIR / f"eval_{ts}.json").write_text(
        json.dumps({"runs": runs, "checks": checks}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
