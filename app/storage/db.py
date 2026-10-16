"""App database (SQLite, file: data/app.sqlite).

Stores things the UI and evaluation need later:
  runs       - one row per question asked (status, tools used, tokens, cost, answer)
  llm_calls  - one row per LLM call inside a run (purpose, tokens, cost, latency)
  reports    - generated pharmacovigilance reports (markdown + citations + PDF path)

Chat memory itself lives in a separate SQLite file (data/memory.sqlite) managed by
LangGraph's checkpointer. See app/agent/graph.py.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from app.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, thread_id TEXT, question TEXT, status TEXT,
    answer TEXT, tools_used TEXT, steps INTEGER DEFAULT 0, model TEXT,
    llm_calls INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0, error TEXT, report_id TEXT,
    started_at REAL, ended_at REAL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, purpose TEXT, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, thinking_tokens INTEGER,
    cost_usd REAL, latency_ms INTEGER, ts REAL
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY, thread_id TEXT, run_id TEXT, title TEXT, markdown TEXT,
    citations TEXT, unverified_citations TEXT, pdf_path TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_thread ON runs(thread_id);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(get_settings().app_db)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def create_run(run_id: str, thread_id: str, question: str, model: str) -> None:
    with connect() as c:
        c.execute("INSERT INTO runs (id, thread_id, question, status, model, started_at) "
                  "VALUES (?,?,?,?,?,?)", (run_id, thread_id, question, "running", model, time.time()))


def finish_run(run_id: str, *, status: str, answer: str, tools_used: list[str], steps: int,
               usage: dict, error: str | None = None, report_id: str | None = None) -> None:
    with connect() as c:
        c.execute(
            "UPDATE runs SET status=?, answer=?, tools_used=?, steps=?, llm_calls=?, input_tokens=?, "
            "output_tokens=?, cost_usd=?, error=?, report_id=?, ended_at=? WHERE id=?",
            (status, answer, json.dumps(tools_used), steps, usage["llm_calls"], usage["input_tokens"],
             usage["output_tokens"], usage["equivalent_cost_usd"], error, report_id, time.time(), run_id))
        c.executemany(
            "INSERT INTO llm_calls (run_id, purpose, model, input_tokens, output_tokens, thinking_tokens, "
            "cost_usd, latency_ms, ts) VALUES (?,?,?,?,?,?,?,?,?)",
            [(run_id, x["purpose"], x["model"], x["input_tokens"], x["output_tokens"],
              x["thinking_tokens"], x["cost_usd"], x["latency_ms"], x["ts"]) for x in usage["calls"]])


def save_report(report: dict) -> None:
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO reports VALUES (?,?,?,?,?,?,?,?,?)",
                  (report["id"], report["thread_id"], report.get("run_id"), report["title"],
                   report["markdown"], json.dumps(report["citations"]),
                   json.dumps(report["unverified_citations"]), report.get("pdf_path"), time.time()))


def get_report(report_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["citations"] = json.loads(d["citations"] or "[]")
    d["unverified_citations"] = json.loads(d["unverified_citations"] or "[]")
    return d


def latest_report_for_thread(thread_id: str, since: float = 0) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT id FROM reports WHERE thread_id=? AND created_at>=? "
                      "ORDER BY created_at DESC LIMIT 1", (thread_id, since)).fetchone()
    return get_report(r["id"]) if r else None


def list_runs(limit: int = 50, thread_id: str | None = None) -> list[dict]:
    q = "SELECT * FROM runs"
    args: tuple = ()
    if thread_id:
        q += " WHERE thread_id=?"
        args = (thread_id,)
    q += " ORDER BY started_at DESC LIMIT ?"
    with connect() as c:
        rows = [dict(r) for r in c.execute(q, args + (limit,)).fetchall()]
    for r in rows:
        r["tools_used"] = json.loads(r["tools_used"] or "[]")
    return rows


def list_threads(limit: int = 30) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT thread_id, MIN(question) AS first_q, COUNT(*) AS runs, SUM(cost_usd) AS cost_usd, "
            "MAX(started_at) AS last_at, "
            "(SELECT question FROM runs r2 WHERE r2.thread_id=runs.thread_id ORDER BY started_at LIMIT 1) AS title "
            "FROM runs GROUP BY thread_id ORDER BY last_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def delete_thread(thread_id: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM llm_calls WHERE run_id IN (SELECT id FROM runs WHERE thread_id=?)", (thread_id,))
        c.execute("DELETE FROM runs WHERE thread_id=?", (thread_id,))
        c.execute("DELETE FROM reports WHERE thread_id=?", (thread_id,))


def stats() -> dict:
    with connect() as c:
        r = c.execute("SELECT COUNT(*) AS runs, COALESCE(SUM(input_tokens),0) AS input_tokens, "
                      "COALESCE(SUM(output_tokens),0) AS output_tokens, COALESCE(SUM(cost_usd),0) AS cost_usd, "
                      "COALESCE(SUM(llm_calls),0) AS llm_calls, "
                      "COALESCE(AVG(CASE WHEN status='completed' THEN 1.0 ELSE 0 END),0) AS completion_rate "
                      "FROM runs WHERE status!='running'").fetchone()
        by_purpose = c.execute("SELECT purpose, COUNT(*) n, SUM(input_tokens) inp, SUM(output_tokens) outp, "
                               "SUM(cost_usd) cost FROM llm_calls GROUP BY purpose").fetchall()
    return {**dict(r), "by_purpose": [dict(x) for x in by_purpose]}
