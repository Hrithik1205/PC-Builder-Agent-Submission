"""Probe several free GitHub Models to see which still have output budget.

Run with: python scripts/probe_github_models.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from src.config import get_settings


CANDIDATES = [
    # Meta / Llama (likely throttled - what user has been using)
    "meta/meta-llama-3.1-8b-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    # OpenAI - separate quota family
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    # Microsoft Phi - separate quota family
    "microsoft/phi-4",
    "microsoft/Phi-3.5-mini-instruct",
    # Mistral - separate quota family
    "mistral-ai/Mistral-large-2411",
    "mistral-ai/mistral-small-2503",
    # Cohere - separate quota family
    "cohere/cohere-command-r-08-2024",
    # xAI
    "xai/grok-3-mini",
]


def probe(token: str, base_url: str, model: str, verify: bool) -> dict:
    """Send one short chat completion. Return latency + output-token count."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with one short paragraph (4-6 sentences)."},
            {"role": "user", "content": "Describe a balanced $1500 gaming PC build."},
        ],
        "temperature": 0.2,
        "max_tokens": 400,  # ask for plenty - server caps if quota is exhausted
    }
    t0 = time.time()
    try:
        with httpx.Client(verify=verify, timeout=30.0) as client:
            resp = client.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
        elapsed_ms = int((time.time() - t0) * 1000)
        if resp.status_code != 200:
            return {"ok": False, "status": resp.status_code,
                    "body_preview": resp.text[:160], "ms": elapsed_ms}
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = data.get("usage") or {}
        return {
            "ok": True,
            "status": 200,
            "ms": elapsed_ms,
            "out_tokens": usage.get("completion_tokens"),
            "in_tokens": usage.get("prompt_tokens"),
            "chars": len(text),
            "preview": text[:160].replace("\n", " "),
        }
    except Exception as e:
        return {"ok": False, "status": None, "error": str(e)[:160], "ms": 0}


def main() -> int:
    s = get_settings()
    token = s.github_token
    if not token:
        print("GITHUB_TOKEN is empty in .env - cannot probe.")
        return 1
    base_url = s.github_models_base_url
    verify = bool(s.data_ssl_verify)
    print(f"Probing {len(CANDIDATES)} models on {base_url}")
    print(f"SSL verify: {verify}\n")
    results = []
    for model in CANDIDATES:
        r = probe(token, base_url, model, verify)
        results.append((model, r))
        if r.get("ok"):
            flag = "OK " if (r.get("out_tokens") or 0) > 50 else "CAP"
            print(
                f"  [{flag}] {model:50s} "
                f"{r.get('chars'):>4} chars / "
                f"{r.get('out_tokens'):>4} tokens / {r.get('ms'):>5} ms"
            )
        else:
            print(
                f"  [ERR] {model:50s} "
                f"status={r.get('status')} {r.get('body_preview') or r.get('error')}"
            )

    print("\n=== Recommendation ===")
    usable = [(m, r) for m, r in results
              if r.get("ok") and (r.get("out_tokens") or 0) > 50]
    if usable:
        usable.sort(key=lambda x: -x[1].get("out_tokens"))
        best, br = usable[0]
        print(f"  Switch to: {best}")
        print(f"  ({br.get('out_tokens')} tokens / {br.get('chars')} chars in {br.get('ms')} ms)")
    else:
        print("  No model returned > 50 tokens. Your overall account is throttled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
