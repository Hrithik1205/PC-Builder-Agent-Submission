"""Probe LLM provider endpoints from the current network.

Reports which providers are reachable, which are DNS-blocked, which are
firewall-blocked, and which need auth (which still means the network path works).
"""
from __future__ import annotations

import socket
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ENDPOINTS = [
    ("HuggingFace API",     "https://huggingface.co/api/whoami-v2"),
    ("HuggingFace Inf API", "https://api-inference.huggingface.co/models/gpt2"),
    ("HuggingFace Router",  "https://router.huggingface.co/v1/models"),
    ("Groq API",            "https://api.groq.com/openai/v1/models"),
    ("OpenRouter",          "https://openrouter.ai/api/v1/models"),
    ("Together AI",         "https://api.together.xyz/v1/models"),
    ("Cerebras",            "https://api.cerebras.ai/v1/models"),
    ("Mistral",             "https://api.mistral.ai/v1/models"),
    ("Cohere",              "https://api.cohere.com/v1/models"),
    ("Google Gemini",       "https://generativelanguage.googleapis.com/v1beta/models"),
    ("Ollama Download",     "https://ollama.com/"),
    ("Ollama Registry",     "https://registry.ollama.ai/v2/"),
    ("ngrok",               "https://ngrok.com"),
    ("Cloudflare Tunnel",   "https://trycloudflare.com"),
]


def classify(name: str, url: str) -> tuple[str, str]:
    host = url.split("//", 1)[1].split("/", 1)[0]
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        return "DNS BLK", "PwC blocks DNS for this host"

    try:
        r = requests.get(url, timeout=8, verify=False)
        return f"OK ({r.status_code})", "reachable" if r.status_code < 400 else "reachable (needs auth)"
    except requests.exceptions.SSLError as e:
        return "SSL ERR", str(e)[:60]
    except requests.exceptions.ConnectionError as e:
        msg = str(e)
        if "Connection refused" in msg:
            return "REFUSED", "TCP refused"
        if "timed out" in msg or "timeout" in msg.lower():
            return "TIMEOUT", "firewall silently dropping"
        return "CONN ERR", msg[:60]
    except requests.exceptions.Timeout:
        return "TIMEOUT", "firewall silently dropping"
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {str(e)[:60]}"


def main():
    print(f"\n{'Service':<22} {'Status':<12} Note")
    print("-" * 80)
    for name, url in ENDPOINTS:
        status, note = classify(name, url)
        print(f"{name:<22} {status:<12} {note}")
    print()


if __name__ == "__main__":
    main()
