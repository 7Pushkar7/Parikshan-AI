"""FastAPI backend.

FastAPI is a Python web framework. It turns Python functions into HTTP endpoints (URLs) that the
browser UI, Postman, or another service can call. It also auto-generates interactive API docs at
http://localhost:8000/docs (Swagger UI).

Run:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agent.service import AgentService
from app.config import BASE_DIR, get_settings
from app.cost import price_for
from app.llm import plain_llm_call
from app.rag import vector_store
from app.storage import db
from app.tools import ALL_TOOLS, TOOL_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("parikshan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: open SQLite memory and build the agent graph."""
    get_settings()
    try:
        async with AgentService.create() as service:
            app.state.agent = service
            app.state.agent_error = None
            log.info("Agent ready with model %s", get_settings().gemini_model)
            yield
    except RuntimeError as exc:  # e.g. missing GEMINI_API_KEY: keep the UI up and show the error
        if "GEMINI_API_KEY" not in str(exc):
            raise
        log.error(str(exc))
        db.init_db()
        app.state.agent = None
        app.state.agent_error = str(exc)
        yield


app = FastAPI(title="Parikshan AI", version="1.0.0", lifespan=lifespan,
              description="Agentic clinical trial & drug safety intelligence API")


# ---------------------------- request / response models ----------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=2000,
                         examples=["Compare efficacy and safety of semaglutide vs tirzepatide for obesity"])
    thread_id: str | None = Field(None, description="Conversation id. Reuse it for follow-up questions.")


class CompareRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=2000)


def _agent(request: Request) -> AgentService:
    if request.app.state.agent is None:
        raise HTTPException(503, request.app.state.agent_error or "Agent not ready")
    return request.app.state.agent


# ---------------------------------- endpoints ----------------------------------

@app.get("/api/health")
async def health(request: Request):
    s = get_settings()
    pin, pout = price_for(s.gemini_model)
    return {"status": "ok" if request.app.state.agent else "missing_api_key",
            "error": request.app.state.agent_error, "model": s.gemini_model, "tier": s.gemini_tier,
            "price_per_1m_tokens_usd": {"input": pin, "output": pout}, "usd_to_inr": s.usd_to_inr,
            "embeddings": s.embeddings, "max_agent_steps": s.max_agent_steps,
            "tools": [{"name": t.name, "label": TOOL_LABELS[t.name],
                       "description": (t.description or "").split("\n")[0]} for t in ALL_TOOLS]}


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """Streams the agent's steps live as Server-Sent Events (SSE): thought, tool_call, tool_result,
    usage, report, final, run_finished. SSE = one long HTTP response that sends small messages."""
    agent = _agent(request)
    thread_id = body.thread_id or uuid.uuid4().hex[:12]

    async def events():
        async for ev in agent.stream(body.message, thread_id):
            if await request.is_disconnected():
                break
            yield {"event": ev["type"], "data": json.dumps(ev, default=str)}

    return EventSourceResponse(events())


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    """Same as the stream endpoint but waits and returns one JSON (handy for Postman / scripts)."""
    return await _agent(request).ask(body.message, body.thread_id or uuid.uuid4().hex[:12])


@app.post("/api/compare")
async def compare_llm_only(body: CompareRequest, request: Request):
    """ONE plain LLM call (no tools, no loop). The UI shows it next to the agent to teach the difference."""
    _agent(request)
    return await plain_llm_call(body.message)


@app.get("/api/threads")
async def threads():
    return db.list_threads()


@app.get("/api/threads/{thread_id}")
async def thread_history(thread_id: str, request: Request):
    return {"thread_id": thread_id, "messages": await _agent(request).history(thread_id),
            "runs": db.list_runs(thread_id=thread_id),
            "sources": list(vector_store.list_sources(thread_id).values())}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str, request: Request):
    await _agent(request).forget(thread_id)
    return {"deleted": thread_id}


@app.get("/api/runs")
async def runs(limit: int = 50):
    return db.list_runs(limit=limit)


@app.get("/api/stats")
async def stats():
    return db.stats()


@app.get("/api/reports/{report_id}")
async def report(report_id: str):
    rep = db.get_report(report_id)
    if not rep:
        raise HTTPException(404, "Report not found")
    return rep


@app.get("/api/reports/{report_id}/pdf")
async def report_pdf(report_id: str):
    rep = db.get_report(report_id)
    if not rep or not rep.get("pdf_path") or not Path(rep["pdf_path"]).exists():
        raise HTTPException(404, "PDF not found")
    safe = "".join(c if c.isalnum() else "_" for c in rep["title"])[:60]
    return FileResponse(rep["pdf_path"], media_type="application/pdf", filename=f"{safe}.pdf")


# The UI: plain HTML + JS served by the same FastAPI app.
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="ui")
