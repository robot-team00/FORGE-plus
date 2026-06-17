"""LLM client abstraction supporting Anthropic, OpenAI-compatible, and mock backends."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import anthropic


SYSTEM_PROMPT = (
    "You are a robotic assembly supervisor. You reason about object fragility "
    "and contact failure modes from mechanical properties and force/contact data. "
    "You always respond with valid JSON matching the requested schema exactly."
)


class LLMClient(ABC):
    @abstractmethod
    def call(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON payload, receive a JSON response."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class AnthropicClient(LLMClient):
    """Frozen Anthropic Claude client — JSON in, JSON out."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 512,
        api_key: str | None = None,
        cache: bool = True,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._cache: dict[str, dict] = {} if cache else None

    def name(self) -> str:
        return self._model

    def call(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        cache_key = json.dumps(user_payload, sort_keys=True)
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        user_text = json.dumps(user_payload, indent=2)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )
        raw = message.content[0].text.strip()

        # Strip markdown fences if model wraps the JSON
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        result = json.loads(raw)
        if self._cache is not None:
            self._cache[cache_key] = result
        return result


class OpenAICompatibleClient(LLMClient):
    """Client for any OpenAI-compatible API (local models, vLLM, etc.)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 512,
        api_key: str = "not-needed",
    ) -> None:
        import openai

        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def name(self) -> str:
        return self._model

    def call(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        user_text = json.dumps(user_payload, indent=2)
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)


class MockLLMClient(LLMClient):
    """Deterministic mock for unit tests — returns sensible hardcoded responses."""

    def __init__(self, budget_n: float = 30.0, recovery_action: str = "retract_and_reapproach") -> None:
        self._budget_n = budget_n
        self._recovery_action = recovery_action

    def name(self) -> str:
        return "mock"

    def call(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        call_type = user_payload.get("call", "")
        if call_type == "set_force_ceiling":
            return {
                "F_max_N": self._budget_n,
                "per_axis_N": {"insertion": self._budget_n, "lateral": self._budget_n * 0.44},
                "confidence": 0.80,
                "rationale": "Mock response: conservative budget for testing.",
            }
        elif call_type == "select_recovery":
            return {
                "action": self._recovery_action,
                "params": {},
                "keep_F_max_N": user_payload.get("F_max_N", self._budget_n),
                "rationale": "Mock response: default recovery action.",
            }
        raise ValueError(f"Unknown call type in MockLLMClient: {call_type}")


def build_client(cfg: dict[str, Any]) -> LLMClient:
    """Factory from config dict."""
    backend = cfg.get("backend", "anthropic")
    if backend == "anthropic":
        return AnthropicClient(
            model=cfg.get("model", "claude-sonnet-4-6"),
            max_tokens=cfg.get("max_tokens", 512),
            cache=cfg.get("cache", True),
        )
    elif backend == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=cfg["base_url"],
            model=cfg["model"],
            max_tokens=cfg.get("max_tokens", 512),
        )
    elif backend == "mock":
        return MockLLMClient(
            budget_n=cfg.get("budget_n", 30.0),
            recovery_action=cfg.get("recovery_action", "retract_and_reapproach"),
        )
    raise ValueError(f"Unknown LLM backend: {backend}")
