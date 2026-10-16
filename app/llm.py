"""Creates the Gemini chat model used everywhere.

Swapping LLM provider later (OpenAI, Claude, Groq) only needs changes in this file,
because the rest of the code talks to LangChain's common `BaseChatModel` interface.
"""
import time
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.cost import UsageTracker

_override: BaseChatModel | None = None  # tests inject a fake model here


def set_model_override(model: BaseChatModel | None) -> None:
    global _override
    _override = model


def get_chat_model(model_name: str | None = None) -> BaseChatModel:
    if _override is not None:
        return _override
    s = get_settings()
    model_name = model_name or s.gemini_model
    if not s.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                           "and put it in the .env file.")
    kwargs = dict(model=model_name, google_api_key=s.gemini_api_key, max_retries=2, timeout=90)
    # Gemini 3+ models are tuned for the default temperature (1.0); Google warns that lowering it
    # can cause looping. So TEMPERATURE only applies to older (2.x) models.
    if model_name.removeprefix("models/").startswith("gemini-2"):
        kwargs["temperature"] = s.temperature
    if s.thinking_budget is not None and model_name.startswith("gemini-2"):
        kwargs["thinking_budget"] = s.thinking_budget
    return ChatGoogleGenerativeAI(**kwargs)


def with_fallback(build: Callable[[BaseChatModel], Runnable]) -> Runnable:
    """Main model, plus an automatic switch to GEMINI_FALLBACK_MODEL if the main one fails
    (503 overloaded, 429 rate limit, ...). `build` adds tools / structured output to a model.
    Each model has its own free-tier quota, so the backup usually still works."""
    s = get_settings()
    primary = build(get_chat_model())
    fb = (s.gemini_fallback_model or "").strip()
    if _override is not None or not fb or fb == s.gemini_model:
        return primary
    return primary.with_fallbacks([build(get_chat_model(fb))])


def text_of(message) -> str:
    """Gemini may return content as a string or a list of blocks; get the plain text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


async def plain_llm_call(question: str) -> dict:
    """ONE plain LLM call, no tools, no loop. Used by the 'LLM vs Agent' tab in the UI."""
    s = get_settings()
    tracker = UsageTracker(s.gemini_model)
    model = with_fallback(lambda m: m)
    t0 = time.perf_counter()
    msg = await model.ainvoke(
        [SystemMessage(content="You are a helpful pharmaceutical research assistant. Answer concisely."),
         HumanMessage(content=question)],
        config={"callbacks": [tracker], "metadata": {"purpose": "plain_llm_call"}},
    )
    return {"answer": text_of(msg), "latency_ms": int((time.perf_counter() - t0) * 1000),
            "usage": tracker.summary()}
