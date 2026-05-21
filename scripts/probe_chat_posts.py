"""Probe whether any mainstream LLM chat endpoint passes PwC's policy filter."""
from __future__ import annotations

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROBES = [
    ("Groq",        "https://api.groq.com/openai/v1/chat/completions", "Bearer fake"),
    ("OpenAI",      "https://api.openai.com/v1/chat/completions", "Bearer fake"),
    ("Anthropic",   "https://api.anthropic.com/v1/messages", "Bearer fake"),
    ("OpenRouter",  "https://openrouter.ai/api/v1/chat/completions", "Bearer fake"),
    ("Together",    "https://api.together.xyz/v1/chat/completions", "Bearer fake"),
    ("Mistral",     "https://api.mistral.ai/v1/chat/completions", "Bearer fake"),
    ("Cerebras",    "https://api.cerebras.ai/v1/chat/completions", "Bearer fake"),
    ("DeepSeek",    "https://api.deepseek.com/v1/chat/completions", "Bearer fake"),
    ("Fireworks",   "https://api.fireworks.ai/inference/v1/chat/completions", "Bearer fake"),
    ("Perplexity",  "https://api.perplexity.ai/chat/completions", "Bearer fake"),
    ("xAI Grok",    "https://api.x.ai/v1/chat/completions", "Bearer fake"),
    ("HF Router",   "https://router.huggingface.co/v1/chat/completions", "Bearer fake"),
    ("Cohere",      "https://api.cohere.com/v2/chat", "Bearer fake"),
    ("GitHub Models","https://models.inference.ai.azure.com/chat/completions", "Bearer fake"),
    ("Azure OpenAI Pub","https://api.azure.com/v1/chat/completions", "Bearer fake"),
    ("AIMLAPI",     "https://api.aimlapi.com/v1/chat/completions", "Bearer fake"),
]


def classify(name: str, url: str, auth: str) -> str:
    payload = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    try:
        r = requests.post(
            url,
            headers={"Authorization": auth, "Content-Type": "application/json"},
            json=payload,
            timeout=10,
            verify=False,
            allow_redirects=False,
        )
        body = r.text[:200].replace("\n", " ")
        if "Access denied" in r.text or "AI-platform-service" in r.text or "pwc.com" in r.text:
            return f"POLICY BLK (PwC AI block)"
        if r.status_code in (200, 400, 401, 403, 404, 422):
            return f"REACHABLE ({r.status_code}) - PwC let it through"
        return f"HTTP {r.status_code} - {body[:80]}"
    except requests.exceptions.SSLError as e:
        return f"SSL ERR: {str(e)[:60]}"
    except requests.exceptions.ConnectionError as e:
        msg = str(e)
        if "Could not resolve" in msg or "name resolution" in msg:
            return "DNS BLK"
        return f"CONN ERR: {msg[:60]}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)[:60]}"


def main():
    print(f"\n{'Provider':<18} Status")
    print("-" * 80)
    for name, url, auth in PROBES:
        status = classify(name, url, auth)
        marker = ""
        if "REACHABLE" in status:
            marker = " <-- USABLE"
        print(f"{name:<18} {status}{marker}")
    print()


if __name__ == "__main__":
    main()
