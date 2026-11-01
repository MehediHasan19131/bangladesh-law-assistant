"""LLM providers: Claude via the Anthropic SDK, or a free local model via Ollama.

Selection (LLM_PROVIDER env var): "anthropic", "ollama", or "auto" (default) which uses Claude when
ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is set and Ollama otherwise.
"""

import os
from dataclasses import dataclass
from typing import Protocol

import requests


@dataclass
class Generation:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str) -> Generation: ...


class ClaudeLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
        self.effort = effort  # grounded Q&A over short contexts doesn't need deep reasoning

    def generate(self, system: str, user: str) -> Generation:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            # on a safety-classifier decline, the API retries the request on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return Generation("The model declined to answer this question.", response.model)
        text = "".join(block.text for block in response.content if block.type == "text")
        return Generation(text.strip(), response.model, response.usage.input_tokens, response.usage.output_tokens)


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "gemma3:4b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def generate(self, system: str, user: str) -> Generation:
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return Generation(data["message"]["content"].strip(), self.model,
                          data.get("prompt_eval_count"), data.get("eval_count"))


def get_llm(provider: str | None = None) -> LLM:
    provider = provider or os.environ.get("LLM_PROVIDER", "auto")
    if provider == "auto":
        has_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        provider = "anthropic" if has_key else "ollama"
    if provider == "anthropic":
        return ClaudeLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"unknown LLM provider {provider!r}")
