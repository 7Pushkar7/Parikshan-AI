"""Benchmark the agent on the drug queries in eval/queries.json.

Metrics (per query, then averaged):
  task_completion   run finished with an answer (and a report, when one was asked for), no crash, within step limit
  tool_recall       share of the expected tools that the agent actually called
  tool_call_success share of tool calls that returned data instead of an error
  citation_grounding share of cited IDs (answer + report) that exist in evidence the agent really fetched
                     (an ungrounded citation = a likely hallucination)
  fact_coverage     share of expected facts (regex) found in the answer + report
Also logs LLM calls, tokens, cost and latency per query.

Usage:
  python -m eval.run_eval                 # all queries
  python -m eval.run_eval --limit 3       # first 3
  python -m eval.run_eval --delay 20      # pause between queries (free-tier rate limits)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

from app.agent.service import AgentService
from app.rag import vector_store
from app.storage import db
from app.tools.analysis import verify_citations

HERE = Path(__file__).parent


def score(q: dict, out: dict) -> dict:
    report_md = ""
    if out["report"]:
        report_md = (db.get_report(out["report"]["report_id"]) or {}).get("markdown", "")
    text = f"{out['answer']}\n{report_md}"
    called = set(out["tools_used"])
    expected = set(q["expected_tools"])
    n_calls = len(out["tools_used"])
    sources = set(vector_store.list_sources(out["thread_id"]))
    verified, unverified = verify_citations(text, sources)
    n_cites = len(verified) + len(unverified)
    facts = [bool(re.search(f, text, re.I)) for f in q["expected_facts"]]
    completed = out["status"] == "completed" and bool(out["answer"]) and (out["report"] is not None or not q["needs_report"])
    return {
        "task_completion": 1.0 if completed else 0.0,
        "tool_recall": len(called & expected) / len(expected) if expected else 1.0,
        "tool_call_success": (n_calls - len(out["tool_errors"])) / n_calls if n_calls else 0.0,
        "citation_grounding": len(verified) / n_cites if n_cites else None,
        "fact_coverage": sum(facts) / len(facts) if facts else None,
        "missing_tools": sorted(expected - called),
        "missing_facts": [f for f, ok in zip(q["expected_facts"], facts) if not ok],
        "unverified_citations": unverified,
    }


def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


async def main(limit: int | None, delay: float, only: str | None):
    queries = json.loads((HERE / "queries.json").read_text())["queries"]
    if only:
        queries = [q for q in queries if q["id"] in only.split(",")]
    queries = queries[:limit] if limit else queries
    rows = []
    async with AgentService.create() as agent:
        for i, q in enumerate(queries, 1):
            thread = f"eval-{q['id']}-{int(time.time())}"
            print(f"[{i}/{len(queries)}] {q['id']}: {q['question'][:70]}...", flush=True)
            out = await agent.ask(q["question"], thread)
            s = score(q, out)
            rows.append({"id": q["id"], "question": q["question"], "status": out["status"], "error": out["error"],
                         "tools_used": out["tools_used"], "steps": out["steps"], "seconds": out["seconds"],
                         "llm_calls": out["usage"]["llm_calls"], "input_tokens": out["usage"]["input_tokens"],
                         "output_tokens": out["usage"]["output_tokens"],
                         "cost_usd": out["usage"]["equivalent_cost_usd"], **s})
            print(f"    status={out['status']} tools={out['tools_used']} cost=${out['usage']['equivalent_cost_usd']:.4f} "
                  f"facts={s['fact_coverage']} grounding={s['citation_grounding']}", flush=True)
            if i < len(queries):
                await asyncio.sleep(delay)

    metrics = ["task_completion", "tool_recall", "tool_call_success", "citation_grounding", "fact_coverage"]
    summary = {m: mean(r[m] for r in rows) for m in metrics}
    summary.update({
        "queries": len(rows),
        "avg_llm_calls": mean(r["llm_calls"] for r in rows),
        "avg_input_tokens": mean(r["input_tokens"] for r in rows),
        "avg_output_tokens": mean(r["output_tokens"] for r in rows),
        "avg_cost_usd": mean(r["cost_usd"] for r in rows),
        "avg_seconds": mean(r["seconds"] for r in rows),
    })
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("\n| id | status | tool recall | tool success | grounding | facts | calls | cost $ |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['id']} | {r['status']} | {r['tool_recall']:.2f} | {r['tool_call_success']:.2f} | "
              f"{r['citation_grounding'] if r['citation_grounding'] is not None else '-'} | "
              f"{r['fact_coverage'] if r['fact_coverage'] is not None else '-'} | {r['llm_calls']} | {r['cost_usd']:.4f} |")
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--delay", type=float, default=15.0)
    ap.add_argument("--only", help="comma-separated ids, e.g. q01,q04")
    a = ap.parse_args()
    asyncio.run(main(a.limit, a.delay, a.only))
