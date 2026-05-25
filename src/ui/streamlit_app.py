"""Streamlit chat UI.

Run with:
    streamlit run src/ui/streamlit_app.py
"""
from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph, make_thread_config
from src.compatibility.engine import check_build, summarize_issues
from src.config import get_settings
from src.data.schemas import Build
from src.logging_setup import configure_logging, current_trace_path


def _init_once():
    """One-time setup per browser session (survives Streamlit reruns)."""
    if st.session_state.get("_app_initialized"):
        return
    configure_logging()
    st.session_state._app_initialized = True
    st.session_state._graph = build_graph(with_memory=True)


def _get_graph():
    _init_once()
    return st.session_state._graph


def _llm_status():
    """Show whether the configured LLM provider is reachable / configured."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "github":
        if not settings.github_token:
            st.error(
                "GitHub Models: no token — create a free PAT at "
                "https://github.com/settings/tokens and add to `.env` as GITHUB_TOKEN"
            )
            return
        st.success(f"GitHub Models: token set ({settings.github_model})")
        return

    if provider == "cerebras":
        if not settings.cerebras_api_key:
            st.error(
                "Cerebras: no API key — get a free key at "
                "https://cloud.cerebras.ai and add to `.env` as CEREBRAS_API_KEY"
            )
            return
        try:
            import urllib.request
            req = urllib.request.Request(
                settings.cerebras_base_url.rstrip("/") + "/models",
                method="GET",
                headers={"Authorization": f"Bearer {settings.cerebras_api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    st.success(f"Cerebras: online ({settings.cerebras_model})")
                    return
        except Exception as e:
            st.error(f"Cerebras: cannot reach API ({type(e).__name__})")
            return

    if provider == "huggingface":
        if not settings.hf_token:
            st.error(
                "HuggingFace: no token — paste a free token from "
                "https://huggingface.co/settings/tokens into `.env` as HF_TOKEN"
            )
            return
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://huggingface.co/api/whoami-v2",
                method="GET",
                headers={"Authorization": f"Bearer {settings.hf_token}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    st.success(f"HuggingFace: online ({settings.hf_model})")
                    return
        except Exception as e:
            st.error(f"HuggingFace: cannot reach API ({type(e).__name__})")
            return

    if provider == "groq":
        if not settings.groq_api_key:
            st.error(
                "Groq: no API key — paste a free key from "
                "https://console.groq.com/keys into your `.env` as GROQ_API_KEY"
            )
            return
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/models",
                method="GET",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    st.success(f"Groq: online ({settings.groq_model})")
                    return
        except Exception as e:
            st.error(f"Groq: cannot reach API ({type(e).__name__})")
            return

    if provider == "ollama":
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{settings.ollama_base_url.rstrip('/')}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    st.success(f"Ollama: running ({settings.ollama_model})")
                    return
        except Exception:
            pass
        st.error(
            "Ollama: not running. Either start Ollama "
            "(`ollama serve`) or switch provider in `.env`."
        )
        return

    # openai / anthropic
    st.info(f"Provider: {provider}")


GREETING = (
    "Hi! I'm your **PC Builder Agent**.\n\n"
    "Tell me what kind of PC you'd like and I'll design a compatible build "
    "from the parts catalog. To get the best recommendation, share:\n\n"
    "- **Use case** - gaming, office, content creation, workstation, home server\n"
    "- **Budget** in USD - a single number (`$1500`) **or a range** (`$1200-$1500`)\n"
    "- **Preferences** - form factor, noise level, AMD vs Intel, peripherals\n\n"
    "Example: *\"Build me a 1440p gaming PC for $1200-$1500, preferably AMD, fairly quiet.\"*\n\n"
    "After I propose a build, you can ask me to revise it - try:\n"
    "- *\"make it cheaper\"*, *\"more storage\"*, *\"quieter\"*\n"
    "- *\"swap the GPU for an NVIDIA card\"*\n"
    "- *\"compare this with a $900 budget\"* (I'll show you what changes)\n"
)


def _ensure_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
    if "history" not in st.session_state:
        st.session_state.history = [("assistant", GREETING)]
    if "build" not in st.session_state:
        st.session_state.build = {}
    if "previous_build" not in st.session_state:
        st.session_state.previous_build = {}
    if "processing" not in st.session_state:
        st.session_state.processing = False


def _render_sidebar():
    settings = get_settings()
    with st.sidebar:
        st.markdown("### PC Builder Agent")
        st.write(f"**Provider:** `{settings.llm_provider}`")
        _model_by_provider = {
            "github": settings.github_model,
            "cerebras": settings.cerebras_model,
            "huggingface": settings.hf_model,
            "groq": settings.groq_model,
            "ollama": settings.ollama_model,
            "openai": settings.openai_model,
            "anthropic": settings.anthropic_model,
        }
        model_label = _model_by_provider.get(settings.llm_provider, "?")
        st.write(f"**Model:** `{model_label}`")
        st.write(f"**Thread:** `{st.session_state.thread_id}`")
        _llm_status()
        st.markdown("---")
        if st.button("Start a new build"):
            st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
            st.session_state.history = [("assistant", GREETING)]
            st.session_state.build = {}
            st.session_state.previous_build = {}
            st.session_state.processing = False
            st.rerun()
        st.markdown("---")
        st.markdown("### Current build")
        build = st.session_state.build
        if not build:
            st.caption("No build yet - send a message to start.")
        else:
            total = 0.0
            rows = []
            for cat, comp in build.items():
                if comp and isinstance(comp, dict):
                    price = float(comp.get("price", 0) or 0)
                    total += price
                    rows.append({
                        "Component": cat,
                        "Part": comp.get("name", "?"),
                        "Price": f"${price:.2f}",
                    })
            if rows:
                st.table(rows)
                st.metric("Total", f"${total:.2f}")
            try:
                issues = check_build(Build(**{k: v for k, v in build.items() if v}))
                if not issues:
                    st.success("Compatibility: PASS")
                else:
                    n_err = sum(1 for i in issues if i.severity == "error")
                    if n_err:
                        st.error(f"Compatibility: {n_err} error(s)")
                    else:
                        st.warning(f"Compatibility: {len(issues)} note(s)")
                    with st.expander("Details"):
                        st.code(summarize_issues(issues))
            except Exception as e:
                st.caption(f"(compatibility check skipped: {e})")

        prev = st.session_state.get("previous_build") or {}
        if prev and build:
            st.markdown("---")
            st.markdown("### Compared to previous build")
            diff_rows = []
            for cat in ("cpu", "motherboard", "memory", "video_card",
                        "storage", "power_supply", "case", "cpu_cooler"):
                old = prev.get(cat) or {}
                nxt = build.get(cat) or {}
                old_name = old.get("name")
                nxt_name = nxt.get("name")
                if old_name == nxt_name and old_name is not None:
                    continue
                old_price = float(old.get("price", 0) or 0)
                nxt_price = float(nxt.get("price", 0) or 0)
                delta = nxt_price - old_price
                diff_rows.append({
                    "Component": cat,
                    "Was": f"{old_name or '-'} (${old_price:.0f})",
                    "Now": f"{nxt_name or '-'} (${nxt_price:.0f})",
                    "Δ": ("+" if delta >= 0 else "") + f"${delta:.0f}",
                })
            if diff_rows:
                st.table(diff_rows)
                old_total = sum(float((c or {}).get("price", 0) or 0)
                                for c in prev.values())
                new_total = sum(float((c or {}).get("price", 0) or 0)
                                for c in build.values())
                st.metric(
                    "Total change",
                    f"${new_total:.2f}",
                    delta=f"{new_total - old_total:+.2f}",
                )
            else:
                st.caption("No component changes vs previous build.")

        st.markdown("---")
        trace = current_trace_path()
        if trace:
            st.caption(f"Trace: `{trace}`")


def _run_agent(user_text: str) -> str:
    """Invoke the graph and return the assistant reply text."""
    graph = _get_graph()
    config = make_thread_config(st.session_state.thread_id)
    state_input = {"messages": [HumanMessage(content=user_text)]}
    result = graph.invoke(state_input, config=config)
    # Track previous build for sidebar diff view (cleared automatically when
    # the new build equals the previous one).
    prev_build = result.get("previous_build")
    if prev_build:
        st.session_state.previous_build = prev_build
    st.session_state.build = result.get("build") or st.session_state.build
    msgs = result.get("messages") or []
    ai_msgs = [m for m in msgs if getattr(m, "type", "") == "ai"]
    if ai_msgs:
        return ai_msgs[-1].content or "(empty response)"
    return result.get("final_response") or "(no response)"


def main():
    st.set_page_config(page_title="PC Builder Agent", page_icon=None, layout="wide")
    _ensure_session()

    try:
        _init_once()
    except Exception as e:
        st.error(f"Failed to start the agent: {e}")
        st.info(
            "Try running `powershell -File scripts/download_data.ps1` first, "
            "then restart Streamlit. On corporate VPN, ensure `DATA_SSL_VERIFY=false` in `.env`."
        )
        return

    _render_sidebar()

    st.title("PC Builder Agent")
    st.caption(
        "Open-source agentic AI that picks compatible PC parts from a fixed "
        "catalog. Type your requirements below."
    )

    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    prompt = st.chat_input(
        "e.g. 'Gaming PC for $1500, 1440p'",
        disabled=st.session_state.processing,
    )

    if prompt and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.history.append(("user", prompt))
        try:
            with st.spinner("Building your PC (first run may take 1-2 min to load data)..."):
                reply = _run_agent(prompt)
            st.session_state.history.append(("assistant", reply))
        except Exception as e:
            st.session_state.history.append(
                ("assistant", f"Agent error: {e}")
            )
        finally:
            st.session_state.processing = False
        st.rerun()


if __name__ == "__main__":
    main()
