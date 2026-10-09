"""How much does the local model have to read, and how long does reading take?

Sends the real copilot request for a few notebook states to Ollama, asks for just 1 output
token (so we time the reading, not the writing), and reports Ollama's own counters.

Run from backend/:  python -m evals.prompt_size [--model qwen3.5:2b]
"""

import argparse
import time

import httpx

from app.config import settings
from app.ollama_client import build_messages, build_tools
from evals.copilot_eval import cleaned, fresh, locked

STATES = {
    "before lock": fresh,
    "after lock (cleaning)": locked,
    "ready to train": cleaned,
}


def measure(session, label: str, model: str) -> None:
    message = "What should I do next?"
    payload = {
        "model": model,
        "messages": build_messages(session, message),
        "tools": build_tools(session),
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        # A unique first line defeats Ollama's prompt cache so every state is measured cold
        "options": {"num_predict": 1, "num_ctx": settings.OLLAMA_CONTEXT, "seed": int(time.time())},
    }
    payload["messages"][0]["content"] = f"[run {time.time()}]\n" + payload["messages"][0]["content"]
    started = time.perf_counter()
    data = httpx.post(f"{settings.OLLAMA_URL}/api/chat", json=payload, timeout=900).json()
    total = time.perf_counter() - started
    tokens = data.get("prompt_eval_count", 0)
    reading = data.get("prompt_eval_duration", 0) / 1e9
    print(f"{label:24} tools={len(payload['tools']):2}  prompt tokens={tokens:5}  "
          f"reading={reading:6.1f}s  ({tokens / reading if reading else 0:5.1f} tokens/s)  total={total:6.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.OLLAMA_MODEL)
    model = parser.parse_args().model
    # Warm up: load the model into memory so loading time isn't counted as reading
    httpx.post(f"{settings.OLLAMA_URL}/api/generate", json={"model": model, "prompt": "hi", "options": {"num_predict": 1}},
               timeout=900)
    for label, setup in STATES.items():
        measure(setup(), label, model)


if __name__ == "__main__":
    main()
