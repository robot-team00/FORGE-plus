"""LLM client abstraction supporting Anthropic, OpenAI-compatible, and mock backends."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any


SYSTEM_PROMPT = (
    "You are a robotic assembly supervisor. You reason about object fragility "
    "and contact failure modes from mechanical properties and force/contact data. "
    "You always respond with valid JSON matching the requested schema exactly."
)


def _strip_code_fences(text: str) -> str:
    """Strip a leading ```/```json markdown fence (and trailing ```) if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a model response into a dict, tolerating fences and surrounding prose."""
    text = _strip_code_fences(text)
    # Fast path: the whole response is valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: decode the first complete JSON object, ignoring any preamble or
    # trailing text. raw_decode stops at the end of the first object, so this is
    # robust to multiple blocks where a greedy `{.*}` regex would fail.
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in model response: {text!r}")


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
        import anthropic

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
        raw = message.content[0].text

        result = parse_json_response(raw)
        if self._cache is not None:
            self._cache[cache_key] = result
        return result


class OpenAICompatibleClient(LLMClient):
    """Client for any OpenAI-compatible API (local models via Ollama, vLLM, llama.cpp, etc.)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 512,
        api_key: str = "not-needed",
        cache: bool = True,
        use_json_mode: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        import openai

        self._model = model
        self._max_tokens = max_tokens
        self._use_json_mode = use_json_mode
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self._cache: dict[str, dict] | None = {} if cache else None

    def name(self) -> str:
        return self._model

    def call(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        cache_key = json.dumps(user_payload, sort_keys=True)
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        user_text = json.dumps(user_payload, indent=2)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,  # greedy decoding: reproducible budgets/recoveries
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        )
        if self._use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content.strip()
                result = self._extract_json(raw)
                if self._cache is not None:
                    self._cache[cache_key] = result
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))

        raise RuntimeError(
            f"Local LLM call failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        return parse_json_response(text)


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



class HeuristicLLMClient(LLMClient):
    """Deterministic, identity-driven stand-in for a frozen reasoning LLM.

    Unlike MockLLMClient (constant budget), this reasons from the *object
    identity* the way a capable LLM would, and from the *force signature* for
    recovery. It lets the full Task-3 pipeline run -- and the budget report be
    meaningful -- with no API key.

    Budget rule (identity only -- never sees F_break): a material-class base
    ceiling, scaled down for brittleness / thin walls and up for thick walls /
    robustness. The point the benchmark probes: a borosilicate glass bowl gets a
    far lower ceiling than a thick stoneware mug, from identity alone.
    """

    _MATERIAL_BASE_N = {
        "glass": 16.0,
        "ceramic": 18.0,
        "resin": 42.0,
        "abs": 45.0,
        "stoneware": 60.0,
        "alumin": 85.0,
        "steel": 110.0,
    }

    def name(self) -> str:
        return "heuristic"

    def call(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        call_type = user_payload.get("call", "")
        if call_type == "set_force_ceiling":
            return self._set_budget(user_payload)
        if call_type == "select_recovery":
            return self._select_recovery(user_payload)
        raise ValueError(f"Unknown call type in HeuristicLLMClient: {call_type}")

    def _set_budget(self, payload: dict[str, Any]) -> dict[str, Any]:
        obj = payload.get("object", {})
        material = str(obj.get("material", "")).lower()
        tags = [str(t).lower() for t in obj.get("geometry_tags", [])]
        cap = float(payload.get("global_hard_cap_N", 120.0))

        base = 50.0
        for key, val in self._MATERIAL_BASE_N.items():
            if key in material:
                base = val
                break

        brittle = "brittle" in tags
        if brittle:
            base *= 0.62
        if "thin_wall" in tags:
            base *= 0.80
        if "curved_rim" in tags:
            base *= 0.90
        if "thick_wall" in tags:
            base *= 1.10
        if any(t.startswith("robust") for t in tags):
            base *= 1.05

        f_max = max(0.0, min(base, cap))
        confidence = 0.85 if (brittle or base < 25) else 0.7
        rationale = (
            f"{material or 'unknown material'}"
            + (", brittle" if brittle else "")
            + f"; identity-only ceiling {f_max:.1f} N"
            + (" (fragile -- keep low; press-harder would break it)" if f_max < 25 else "")
        )
        return {
            "F_max_N": round(f_max, 1),
            "per_axis_N": {"insertion": round(f_max, 1), "lateral": round(f_max * 0.4, 1)},
            "confidence": confidence,
            "rationale": rationale,
        }

    def _select_recovery(self, payload: dict[str, Any]) -> dict[str, Any]:
        sig = payload.get("signature", {})
        attempt = int(payload.get("attempt", 0))
        f_max = payload.get("F_max_N", 0.0)
        lateral_bias = str(sig.get("lateral_bias", "none"))
        torque_z = abs(float(sig.get("torque_z_Nm", 0.0)))
        slip = int(sig.get("slip_events", 0))
        peak_lat = float(sig.get("peak_lateral_N", 0.0))

        if attempt >= 4:
            action, why = "abort", "repeated failures; abort rather than risk the part"
        elif lateral_bias != "none" or peak_lat > 2.0:
            action, why = "rotate_align", "lateral bias at contact (edge-load) -> realign to surface"
        elif torque_z > 0.3:
            action, why = "rotate_align", "EE torque (tipping) -> realign before re-placing"
        elif slip > 0:
            action, why = "regrasp", "grasp slip detected -> re-centre the part"
        else:
            action, why = "retract_and_reapproach", "lift off and re-approach within the same budget"

        return {"action": action, "params": {}, "keep_F_max_N": f_max, "rationale": why}


def build_client(cfg: dict[str, Any]) -> LLMClient:
    """Factory from config dict."""
    backend = cfg.get("backend", "anthropic")
    if backend == "anthropic":
        return AnthropicClient(
            model=cfg.get("model", "claude-sonnet-4-6"),
            max_tokens=cfg.get("max_tokens", 512),
            cache=cfg.get("cache", True),
        )
    elif backend in ("openai_compatible", "local"):
        # "local" is a convenience alias for openai_compatible that defaults to
        # Ollama on localhost with Qwen2.5 and JSON-mode off (Ollama doesn't
        # reliably honour response_format; we fall back to regex extraction).
        return OpenAICompatibleClient(
            base_url=cfg.get("base_url", "http://localhost:11434/v1"),
            model=cfg.get("model", "llama3.1:8b"),
            max_tokens=cfg.get("max_tokens", 512),
            cache=cfg.get("cache", True),
            use_json_mode=cfg.get("use_json_mode", backend == "openai_compatible"),
            max_retries=cfg.get("max_retries", 3),
            retry_delay=cfg.get("retry_delay", 2.0),
        )
    elif backend == "mock":
        return MockLLMClient(
            budget_n=cfg.get("budget_n", 30.0),
            recovery_action=cfg.get("recovery_action", "retract_and_reapproach"),
        )
    elif backend == "heuristic":
        return HeuristicLLMClient()
    raise ValueError(f"Unknown LLM backend: {backend}")
