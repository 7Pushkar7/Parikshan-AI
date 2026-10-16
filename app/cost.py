"""Token + cost tracking for every LLM call.

How LLM billing works:
  cost = input_tokens / 1M * input_price  +  output_tokens / 1M * output_price
  - input tokens  = everything sent: system prompt + chat history + tool results + question
  - output tokens = everything generated, INCLUDING Gemini "thinking" tokens

Prices: Gemini Developer API, Standard tier, USD per 1M text tokens
(https://ai.google.dev/gemini-api/docs/pricing, checked 16 Oct 2026). Prices change, so the
table is easy to edit, or set PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M in .env.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import get_settings

# model prefix -> (input $/1M, output $/1M). Longest matching prefix wins.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),      # no longer offered to new users
    "gemini-2.5-pro": (1.25, 10.00),       # prompts <= 200k tokens
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro": (2.00, 12.00),       # prompts <= 200k tokens
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),      # through 31 Dec 2026
    "gemini-3.8-flash": (0.75, 3.75),      # through 31 Dec 2026
}
DEFAULT_PRICE = (0.75, 3.75)  # gemini-3.8-flash


def price_for(model: str) -> tuple[float, float]:
    s = get_settings()
    if s.price_input_per_m is not None and s.price_output_per_m is not None:
        return s.price_input_per_m, s.price_output_per_m
    name = model.removeprefix("models/")
    matches = [k for k in PRICING if name.startswith(k)]
    return PRICING[max(matches, key=len)] if matches else DEFAULT_PRICE


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return input_tokens / 1_000_000 * pin + output_tokens / 1_000_000 * pout


@dataclass
class LLMCallRecord:
    purpose: str
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float
    latency_ms: int
    ts: float = field(default_factory=time.time)


class UsageTracker(BaseCallbackHandler):
    """LangChain callback: fires on every LLM call made during one agent run
    (the agent's own reasoning steps AND LLM calls made inside tools)."""

    def __init__(self, model: str):
        self.model = model
        self.calls: list[LLMCallRecord] = []
        self._starts: dict[UUID, tuple[float, str]] = {}

    def _purpose(self, metadata: dict | None, tags: list[str] | None) -> str:
        metadata = metadata or {}
        if metadata.get("purpose"):
            return metadata["purpose"]
        if metadata.get("langgraph_node") == "agent":
            return "agent_reasoning"
        return (tags or ["llm_call"])[0]

    def on_chat_model_start(self, serialized: dict, messages: Any, *, run_id: UUID,
                            tags: list[str] | None = None, metadata: dict | None = None,
                            **kwargs: Any) -> None:
        self._starts[run_id] = (time.perf_counter(), self._purpose(metadata, tags))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start, purpose = self._starts.pop(run_id, (time.perf_counter(), "llm_call"))
        usage: dict = {}
        model = self.model
        try:
            msg = response.generations[0][0].message  # type: ignore[attr-defined]
            usage = msg.usage_metadata or {}
            model = (msg.response_metadata or {}).get("model_name") or self.model
        except (IndexError, AttributeError):
            usage = (response.llm_output or {}).get("usage_metadata", {}) or {}
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        think = int((usage.get("output_token_details") or {}).get("reasoning", 0) or 0)
        self.calls.append(LLMCallRecord(
            purpose=purpose, model=model, input_tokens=inp, output_tokens=out,
            thinking_tokens=think, cost_usd=cost_usd(model, inp, out),
            latency_ms=int((time.perf_counter() - start) * 1000)))

    def summary(self) -> dict:
        s = get_settings()
        inp = sum(c.input_tokens for c in self.calls)
        out = sum(c.output_tokens for c in self.calls)
        usd = sum(c.cost_usd for c in self.calls)
        pin, pout = price_for(self.model)
        return {
            "model": self.model,
            "tier": s.gemini_tier,
            "llm_calls": len(self.calls),
            "input_tokens": inp,
            "output_tokens": out,
            "thinking_tokens": sum(c.thinking_tokens for c in self.calls),
            "equivalent_cost_usd": round(usd, 6),
            "billed_cost_usd": 0.0 if s.gemini_tier == "free" else round(usd, 6),
            "equivalent_cost_inr": round(usd * s.usd_to_inr, 4),
            "price_per_1m": {"input": pin, "output": pout},
            "calls": [asdict(c) for c in self.calls],
        }
