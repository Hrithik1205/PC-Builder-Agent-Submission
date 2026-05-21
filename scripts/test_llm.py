"""Quick end-to-end test that the configured LLM provider is online."""
from __future__ import annotations

import os
import ssl
import time

# Tell the SSL stack to skip cert verification (PwC corporate VPN intercepts TLS).
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.llm.providers import get_chat_model


def main():
    s = get_settings()
    print(f"Provider: {s.llm_provider}")
    if s.llm_provider == "groq":
        print(f"Model:    {s.groq_model}")
        print(f"Key set:  {bool(s.groq_api_key)}")
    elif s.llm_provider == "huggingface":
        print(f"Model:    {s.hf_model}")
        print(f"Token set: {bool(s.hf_token)}")
    elif s.llm_provider == "ollama":
        print(f"Model:    {s.ollama_model}")
        print(f"URL:      {s.ollama_base_url}")
    print()
    print("Pinging LLM...")

    llm = get_chat_model()
    t0 = time.time()
    resp = llm.invoke([
        SystemMessage(content="You are a concise assistant. Reply in one short sentence."),
        HumanMessage(content="Say 'hello from <model_name>' and nothing else."),
    ])
    elapsed = time.time() - t0
    text = resp.content if hasattr(resp, "content") else str(resp)
    print(f"Latency: {elapsed:.2f}s")
    print(f"Reply:   {text}")
    print()
    print("SUCCESS - LLM is online.")


if __name__ == "__main__":
    main()
