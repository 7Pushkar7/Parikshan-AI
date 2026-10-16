"""Data-fetching tools. Each tool:
  1. calls a free public API,
  2. stores the FULL text in the vector DB (for RAG later),
  3. returns a SHORT summary to the agent (keeps input tokens, and cost, low).

The docstring of each @tool is what Gemini reads to decide WHEN to call it,
so docstrings are written as instructions to the model.
"""
from __future__ import annotations

import json
import re

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.clients import clinical_trials, openfda, pubmed
from app.clients.http import ExternalAPIError
from app.rag import vector_store

MAX_TOOL_CHARS = 6000


def thread_of(config: RunnableConfig | None) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id", "default")


def compact(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= MAX_TOOL_CHARS else s[:MAX_TOOL_CHARS] + '..."(truncated)"'


def _err(tool_name: str, exc: Exception) -> str:
    return compact({"error": f"{tool_name} failed: {exc}",
                    "hint": "Retry once with simpler terms, or continue with other sources."})


@tool
async def search_pubmed(query: str, max_results: int = 5, config: RunnableConfig = None) -> str:
    """Search PubMed for published biomedical papers (trial results, meta-analyses, safety studies).
    Use for efficacy evidence, published results and literature on adverse effects.
    `query` supports PubMed syntax, e.g. 'semaglutide AND obesity AND randomized controlled trial[pt]'.
    Returns PMIDs, titles, journals, years and short abstracts. Cite as [PMID:<id>]."""
    try:
        res = await pubmed.search(query, max(1, min(max_results, 10)))
    except ExternalAPIError as exc:
        return _err("search_pubmed", exc)
    tid = thread_of(config)
    for a in res["articles"]:
        vector_store.add_document(
            tid, f"PMID:{a['pmid']}", "pubmed", a["title"], a["url"],
            f"{a['title']}. {a['journal']} ({a['year']}). Types: {', '.join(a['publication_types'])}. "
            f"{a['abstract']}")
    return compact({
        "query": query, "total_found": res["total_found"],
        "articles": [{"cite_as": f"PMID:{a['pmid']}", "title": a["title"], "journal": a["journal"],
                      "year": a["year"], "types": a["publication_types"][:3],
                      "abstract": a["abstract"][:600]} for a in res["articles"]],
    })


@tool
async def search_clinical_trials(drug: str, condition: str | None = None, status: str | None = None,
                                 phase: str | None = None, max_results: int = 5,
                                 config: RunnableConfig = None) -> str:
    """Search ClinicalTrials.gov for registered clinical trials of a drug.
    Args: drug (intervention name), optional condition, optional status
    (RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING, TERMINATED), optional phase (PHASE1..PHASE4).
    Returns NCT IDs, status, phase, enrollment, sponsor, primary outcomes, and whether results are posted.
    Cite as [NCT########]. Call get_trial_details for eligibility criteria and posted adverse events."""
    try:
        res = await clinical_trials.search(drug, condition, status, phase, max(1, min(max_results, 10)))
    except ExternalAPIError as exc:
        return _err("search_clinical_trials", exc)
    tid = thread_of(config)
    for t in res["trials"]:
        vector_store.add_document(
            tid, t["nct_id"], "clinical_trial", t["title"], t["url"],
            f"{t['title']}. Status: {t['status']}. Phases: {', '.join(t['phases'])}. "
            f"Enrollment: {t['enrollment']}. Sponsor: {t['sponsor']}. Conditions: {', '.join(t['conditions'])}. "
            f"Interventions: {'; '.join(t['interventions'])}. Primary outcomes: {'; '.join(t['primary_outcomes'])}. "
            f"Start {t['start_date']}, completion {t['completion_date']}. Summary: {t['brief_summary']}")
    for t in res["trials"]:
        t["brief_summary"] = t["brief_summary"][:250]
        t["cite_as"] = t.pop("nct_id")
        t.pop("url", None)
    return compact(res)


@tool
async def get_trial_details(nct_id: str, config: RunnableConfig = None) -> str:
    """Get full details of ONE trial by NCT ID: eligibility criteria (inclusion/exclusion, age, sex),
    primary and secondary outcomes, and, if the sponsor posted results, the serious and other
    adverse events per study arm with % of participants affected. Use after search_clinical_trials."""
    if not re.fullmatch(r"NCT\d{8}", nct_id.strip().upper()):
        return compact({"error": f"'{nct_id}' is not a valid NCT ID (format NCT########)."})
    try:
        d = await clinical_trials.get_study(nct_id)
    except ExternalAPIError as exc:
        return _err("get_trial_details", exc)
    ae_text = ""
    if d["adverse_events"]:
        rows = d["adverse_events"]["serious"] + d["adverse_events"]["other"]
        ae_text = " Posted adverse events: " + "; ".join(
            f"{r['term']} in {r['group']}: {r['affected']}/{r['at_risk']} ({r['pct']}%)" for r in rows)
    vector_store.add_document(
        thread_of(config), d["nct_id"], "clinical_trial", d["title"], d["url"],
        f"{d['title']}. Status {d['status']}, phases {', '.join(d['phases'])}, enrollment {d['enrollment']}. "
        f"Primary outcomes: {'; '.join(d['primary_outcomes'])}. Secondary outcomes: "
        f"{'; '.join(d['secondary_outcomes'])}. Eligibility: sex {d['eligibility']['sex']}, age "
        f"{d['eligibility']['minimum_age']} to {d['eligibility']['maximum_age'] or 'no limit'}. "
        f"Criteria: {d['eligibility']['criteria']}.{ae_text}")
    d["eligibility"]["criteria"] = d["eligibility"]["criteria"][:1200]
    d["cite_as"] = d.pop("nct_id")
    d.pop("brief_summary", None)
    return compact(d)


@tool
async def get_drug_safety_data(drug: str, config: RunnableConfig = None) -> str:
    """Get post-marketing safety data for a drug from the US FDA (openFDA):
    total FAERS adverse event reports, serious and death reports, the most reported reactions,
    and the official label's boxed warning / warnings. Use the GENERIC name (e.g. 'semaglutide').
    Cite FAERS data as [FAERS:<drug>] and label text as [LABEL:<drug>].
    FAERS counts are voluntary reports, not incidence rates."""
    try:
        ae = await openfda.adverse_events(drug)
        lbl = await openfda.label(drug)
    except ExternalAPIError as exc:
        return _err("get_drug_safety_data", exc)
    tid = thread_of(config)
    key = drug.strip().lower()
    if ae["total_reports"]:
        vector_store.add_document(
            tid, f"FAERS:{key}", "faers", f"FDA FAERS adverse event reports for {drug}", ae["source_url"],
            f"FDA FAERS data for {drug}: {ae['total_reports']} total reports, {ae['serious_reports']} serious, "
            f"{ae['death_reports']} with death outcome. Most reported reactions: "
            + "; ".join(f"{r['reaction']} ({r['reports']} reports, {r['share_pct']}%)" for r in ae["top_reactions"])
            + f". Caveat: {ae['caveat']}")
    if lbl:
        vector_store.add_document(
            tid, f"LABEL:{key}", "fda_label", f"FDA drug label for {drug}", lbl["url"],
            f"FDA label for {drug} (brands {', '.join(lbl['brand_names'])}). Boxed warning: "
            f"{lbl['boxed_warning'] or 'none'}. Warnings: {lbl['warnings']}. Adverse reactions: "
            f"{lbl['adverse_reactions']}. Contraindications: {lbl['contraindications']}")
    return compact({
        "faers": {**ae, "cite_as": f"FAERS:{key}"},
        "label": ({"cite_as": f"LABEL:{key}", "brand_names": lbl["brand_names"],
                   "boxed_warning": lbl["boxed_warning"][:700], "warnings": lbl["warnings"][:700]}
                  if lbl else "No FDA label found"),
    })
