"""Runs the agent for one question and turns LangGraph's stream into simple UI events.

Event types sent to the UI (Server-Sent Events):
  run_started  {run_id, thread_id}
  thought      {step, text}                    - the agent's reasoning sentence (THINK)
  tool_call    {step, id, name, label, args}   - the agent asks to run a tool (ACT)
  tool_result  {step, id, name, ok, preview, ms} - what the tool returned (OBSERVE)
  usage        {...}                           - running token / cost totals
  report       {report_id, title, ...}         - a PDF report was generated
  final        {answer}
  error        {message}
  run_finished {status, usage, tools_used, steps}
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from app.agent.graph import build_graph
from app.config import get_settings
from app.cost import UsageTracker
from app.llm import text_of
from app.rag import vector_store
from app.storage import db
from app.tools import TOOL_LABELS


class AgentService:
    def __init__(self, graph):
        self.graph = graph

    @classmethod
    @asynccontextmanager
    async def create(cls, memory_path: str | None = None):
        """Opens the SQLite memory (checkpointer) and builds the graph."""
        settings = get_settings()
        db.init_db()
        async with AsyncSqliteSaver.from_conn_string(str(memory_path or settings.memory_db)) as saver:
            yield cls(build_graph(checkpointer=saver))

    async def stream(self, question: str, thread_id: str) -> AsyncIterator[dict]:
        settings = get_settings()
        run_id = db.new_id()
        tracker = UsageTracker(settings.gemini_model)
        config = {
            "configurable": {"thread_id": thread_id, "run_id": run_id},
            "callbacks": [tracker],
            "recursion_limit": settings.max_agent_steps * 2 + 1,
        }
        db.create_run(run_id, thread_id, question, settings.gemini_model)
        started = time.time()
        yield {"type": "run_started", "run_id": run_id, "thread_id": thread_id}

        step, tools_used, tool_started = 0, [], {}
        final_answer, status, error, report_id = "", "completed", None, None
        try:
            async for update in self.graph.astream(
                    {"messages": [HumanMessage(content=question)]}, config, stream_mode="updates"):
                for node, payload in update.items():
                    for msg in (payload or {}).get("messages", []):
                        if node == "agent" and isinstance(msg, AIMessage):
                            step += 1
                            text = text_of(msg).strip()
                            if msg.tool_calls:
                                if text:
                                    yield {"type": "thought", "step": step, "text": text}
                                for tc in msg.tool_calls:
                                    tool_started[tc["id"]] = time.perf_counter()
                                    tools_used.append(tc["name"])
                                    yield {"type": "tool_call", "step": step, "id": tc["id"],
                                           "name": tc["name"], "label": TOOL_LABELS.get(tc["name"], tc["name"]),
                                           "args": tc["args"]}
                            else:
                                final_answer = text
                        elif node == "tools" and isinstance(msg, ToolMessage):
                            content = text_of(msg)
                            ok = msg.status != "error" and '"error"' not in content[:40]
                            ms = int((time.perf_counter() - tool_started.get(msg.tool_call_id, time.perf_counter())) * 1000)
                            yield {"type": "tool_result", "step": step, "id": msg.tool_call_id,
                                   "name": msg.name, "ok": ok, "preview": content[:1500], "chars": len(content),
                                   "ms": ms}
                            if msg.name == "write_report" and ok:
                                try:
                                    rid = json.loads(content).get("report_id")
                                except json.JSONDecodeError:
                                    rid = None
                                rep = db.get_report(rid) if rid else None
                                if rep:
                                    report_id = rid
                                    yield {"type": "report", "report_id": rid, "title": rep["title"],
                                           "markdown": rep["markdown"], "citations": rep["citations"],
                                           "unverified_citations": rep["unverified_citations"]}
                yield {"type": "usage", **_light(tracker.summary())}
        except GraphRecursionError:
            status = "max_steps"
            final_answer = final_answer or (f"I stopped after {settings.max_agent_steps} steps without finishing. "
                                            "Try a narrower question.")
        except Exception as exc:  # noqa: BLE001  (show any failure to the user instead of crashing)
            status, error = "error", f"{type(exc).__name__}: {exc}"
            yield {"type": "error", "message": error}

        if final_answer:
            yield {"type": "final", "answer": final_answer}
        usage = tracker.summary()
        db.finish_run(run_id, status=status, answer=final_answer, tools_used=tools_used, steps=step,
                      usage=usage, error=error, report_id=report_id)
        yield {"type": "run_finished", "run_id": run_id, "status": status, "usage": usage,
               "tools_used": tools_used, "steps": step, "seconds": round(time.time() - started, 1)}

    async def ask(self, question: str, thread_id: str) -> dict:
        """Non-streaming version: run to the end and return everything (used by API + evaluation)."""
        events = [e async for e in self.stream(question, thread_id)]
        done = events[-1]
        return {
            "run_id": done["run_id"], "thread_id": thread_id, "status": done["status"],
            "answer": next((e["answer"] for e in events if e["type"] == "final"), ""),
            "tools_used": done["tools_used"], "steps": done["steps"], "seconds": done["seconds"],
            "usage": done["usage"],
            "report": next(({k: e[k] for k in ("report_id", "title", "citations", "unverified_citations")}
                            for e in events if e["type"] == "report"), None),
            "tool_errors": [e["name"] for e in events if e["type"] == "tool_result" and not e["ok"]],
            "error": next((e["message"] for e in events if e["type"] == "error"), None),
            "trace": [e for e in events if e["type"] in ("thought", "tool_call", "tool_result")],
        }

    async def history(self, thread_id: str) -> list[dict]:
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        out = []
        for m in (state.values or {}).get("messages", []):
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": text_of(m)})
            elif isinstance(m, AIMessage) and not m.tool_calls and text_of(m).strip():
                out.append({"role": "assistant", "content": text_of(m)})
        return out

    async def forget(self, thread_id: str) -> None:
        await self.graph.checkpointer.adelete_thread(thread_id)
        db.delete_thread(thread_id)
        vector_store.delete_thread(thread_id)


def _light(summary: dict) -> dict:
    s = dict(summary)
    s["last_call"] = s["calls"][-1] if s["calls"] else None
    s.pop("calls")
    return s
