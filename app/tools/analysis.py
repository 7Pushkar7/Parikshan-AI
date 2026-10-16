"""Analysis tools that use the LLM themselves:
  extract_entities -> NLP entity extraction with structured (JSON) output
  write_report     -> RAG report engine: retrieve evidence -> generate cited report -> verify citations -> PDF
"""
from __future__ import annotations

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import text_of, with_fallback
from app.pdf import build_report_pdf
from app.rag import vector_store
from app.storage import db
from app.tools.research import compact, thread_of

# --------------------------------------------------------------------------------------
# 1. Entity extraction
# --------------------------------------------------------------------------------------


class Outcome(BaseModel):
    measure: str = Field(description="What was measured, e.g. 'change in body weight at week 68'")
    result: str = Field(description="The reported result with numbers if stated, else 'not reported'")


class SourceEntities(BaseModel):
    source_id: str = Field(description="The source ID exactly as given, e.g. PMID:123 or NCT01234567")
    drugs: list[str] = Field(default_factory=list, description="Drug names (generic or brand)")
    dosages: list[str] = Field(default_factory=list, description="Doses with units and schedule, e.g. '2.4 mg once weekly'")
    population: str = Field(default="", description="Who was studied, e.g. 'adults with BMI>=30 without diabetes, n=1961'")
    outcomes: list[Outcome] = Field(default_factory=list)
    adverse_events: list[str] = Field(default_factory=list, description="Adverse events mentioned, with % if stated")
    safety_signals: list[str] = Field(default_factory=list, description="Serious or notable safety concerns, warnings")


class ExtractionResult(BaseModel):
    sources: list[SourceEntities]


EXTRACT_PROMPT = """You are a biomedical information extraction system.
Extract entities from each source below. Only use facts stated in the text. Never guess numbers.
If something is not stated, leave the list empty or write 'not reported'."""


@tool
async def extract_entities(source_ids: list[str] | None = None, config: RunnableConfig = None) -> str:
    """Run NLP entity extraction on sources ALREADY fetched in this conversation.
    Extracts drug names, dosages, study population, outcomes (with numbers), adverse events and
    safety signals as structured JSON. Pass source IDs like ['PMID:12345', 'NCT01234567'];
    leave empty to process up to 8 fetched sources. Use before comparing trials."""
    tid = thread_of(config)
    available = vector_store.list_sources(tid)
    ids = [s for s in (source_ids or []) if s in available] or list(available)[:8]
    if not ids:
        return compact({"error": "No sources fetched yet. Call a search tool first."})
    blocks = []
    for sid in ids[:8]:
        text = " ".join(c["text"] for c in vector_store.get_source(tid, sid))[:4000]
        blocks.append(f"### SOURCE {sid}\n{text}")
    model = with_fallback(lambda m: m.with_structured_output(ExtractionResult))
    result: ExtractionResult = await model.ainvoke(
        [SystemMessage(content=EXTRACT_PROMPT), HumanMessage(content="\n\n".join(blocks))],
        config={**(config or {}), "metadata": {**((config or {}).get("metadata") or {}),
                                                "purpose": "entity_extraction"}},
    )
    return compact(result.model_dump() if result else {"sources": []})


# --------------------------------------------------------------------------------------
# 2. RAG report writer
# --------------------------------------------------------------------------------------

CITATION_RE = re.compile(r"(PMID:\s*\d+|NCT\d{8}|FAERS:[a-z0-9 \-]+|LABEL:[a-z0-9 \-]+)", re.I)

REPORT_PROMPT = """You are a senior pharmacovigilance and clinical evidence analyst writing for a
medical affairs team. Write a clear, well-structured report in Markdown that answers the REQUEST.

EVIDENCE RULES
- Use ONLY the evidence in the CONTEXT. No outside knowledge.
- Cite sources with their IDs in square brackets exactly as given, e.g. [NCT03548935], [PMID:33567185],
  [FAERS:semaglutide], [LABEL:semaglutide]. Put citations at the END of the sentence.
  Use at most 2 IDs per sentence: pick the strongest ones.
- Never invent numbers, trial names or IDs. If evidence is missing or weak, say so plainly.
- FAERS counts are voluntary reports, not incidence rates. Say this once when you use them.

STYLE RULES
- 500 to 800 words in total. Plain, precise language. Short sentences.
- Bullets: one idea each, at most 2 lines, start with a short **bold label:** then the finding.
- Give numbers with context: "-14.9% vs -2.4% with placebo at week 68" rather than "large reduction".
- Table cells: at most 12 words. The Source column holds only IDs in brackets.
- No HTML, no horizontal rules, no emojis, no references section (it is added automatically).

STRUCTURE (use exactly these headings)
# <Drug or topic>: <what the report covers>
## Executive summary
4 bullets: efficacy, safety, key warning, overall conclusion.
## Trial evidence
1 short paragraph, then a table with columns: Trial | Phase | N | Population | Key result | Source
(at most 6 rows, most important trials first).
## Safety profile
### In clinical trials
2 to 4 bullets with rates where available.
### Real-world reports (FAERS and FDA label)
2 to 4 bullets.
## Signals to watch
2 to 4 bullets.
## Evidence gaps
2 to 3 bullets.
"""


def normalize_citation(raw: str) -> str:
    raw = raw.strip()
    if raw.upper().startswith(("NCT", "PMID")):
        return re.sub(r"\s+", "", raw).upper()
    prefix, rest = raw.split(":", 1)
    return f"{prefix.upper()}:{rest.strip().lower()}"


def verify_citations(markdown_text: str, allowed: set[str]) -> tuple[list[str], list[str]]:
    """Return (verified, unverified) citation IDs. Unverified = cited but never fetched = possible hallucination."""
    allowed_l = {a.lower() for a in allowed}
    found: list[str] = []
    for m in CITATION_RE.findall(markdown_text):
        cid = normalize_citation(m)
        if cid not in found:
            found.append(cid)
    verified = [c for c in found if c.lower() in allowed_l]
    unverified = [c for c in found if c.lower() not in allowed_l]
    return verified, unverified


@tool
async def write_report(request: str, config: RunnableConfig = None) -> str:
    """Generate the final source-cited pharmacovigilance / evidence report (also exported as PDF)
    from ALL evidence fetched in this conversation, using retrieval (RAG).
    Call this LAST, after gathering trials, literature and safety data, when the user wants a
    report, summary, comparison or safety review. `request` = what the report should cover."""
    s = get_settings()
    tid = thread_of(config)
    sources = vector_store.list_sources(tid)
    if not sources:
        return compact({"error": "No evidence fetched yet. Search trials/PubMed/FDA first."})

    hits = vector_store.search(tid, request, k=s.rag_top_k)
    # Always include safety sources: they matter for pharmacovigilance even if similarity is lower.
    have = {h["source_id"] for h in hits}
    for sid, meta in sources.items():
        if meta["source_type"] in ("faers", "fda_label") and sid not in have:
            hits += vector_store.get_source(tid, sid)[:1]
    context = "\n\n".join(f"[{h['source_id']}] {h['title']}\n{h['text']}" for h in hits)

    model = with_fallback(lambda m: m)
    msg = await model.ainvoke(
        [SystemMessage(content=REPORT_PROMPT),
         HumanMessage(content=f"REQUEST: {request}\n\nCONTEXT:\n{context}")],
        config={**(config or {}), "metadata": {**((config or {}).get("metadata") or {}),
                                                "purpose": "report_generation"}},
    )
    body = text_of(msg).strip()
    verified, unverified = verify_citations(body, set(sources))

    title_match = re.search(r"^#\s+(.+)$", body, re.M)
    title = title_match.group(1).strip() if title_match else request[:80]
    citations = []
    for cid in verified:
        meta = sources.get(cid) or next((v for k, v in sources.items() if k.lower() == cid.lower()), {})
        citations.append({"id": cid, "title": meta.get("title", ""), "url": meta.get("url", ""),
                          "source_type": meta.get("source_type", "")})
    report_id = db.new_id()
    used_model = (getattr(msg, "response_metadata", None) or {}).get("model_name") or s.gemini_model
    pdf_path = build_report_pdf(body, s.reports_dir / f"{report_id}.pdf", citations=citations,
                                unverified=unverified, model=used_model, fallback_title=title)
    run_id = ((config or {}).get("configurable") or {}).get("run_id")
    db.save_report({"id": report_id, "thread_id": tid, "run_id": run_id, "title": title, "markdown": body,
                    "citations": citations,
                    "unverified_citations": unverified, "pdf_path": str(pdf_path)})
    return compact({"report_id": report_id, "title": title, "verified_citations": len(verified),
                    "unverified_citations": unverified, "evidence_chunks_used": len(hits),
                    "preview": body[:700],
                    "note": "Report saved and shown to the user with a PDF download. Summarise key findings briefly."})
