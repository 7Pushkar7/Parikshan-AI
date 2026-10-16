"""ClinicalTrials.gov REST API v2 (free, no key needed).

  GET /api/v2/studies            -> search trials
  GET /api/v2/studies/{NCT ID}   -> one full trial record (incl. results + adverse events if posted)
Docs: https://clinicaltrials.gov/data-api/api
"""
from app.clients.http import get

BASE = "https://clinicaltrials.gov/api/v2"

VALID_STATUS = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED",
                "NOT_YET_RECRUITING", "WITHDRAWN", "SUSPENDED", "ENROLLING_BY_INVITATION"}


def _g(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return d if d is not None else default


def summarize_study(study: dict) -> dict:
    """Flatten the deeply nested v2 JSON into the fields the agent needs."""
    p = study.get("protocolSection", {})
    nct = _g(p, "identificationModule", "nctId", default="")
    return {
        "nct_id": nct,
        "title": _g(p, "identificationModule", "briefTitle", default=""),
        "status": _g(p, "statusModule", "overallStatus", default=""),
        "phases": _g(p, "designModule", "phases", default=[]),
        "study_type": _g(p, "designModule", "studyType", default=""),
        "enrollment": _g(p, "designModule", "enrollmentInfo", "count"),
        "start_date": _g(p, "statusModule", "startDateStruct", "date", default=""),
        "completion_date": _g(p, "statusModule", "completionDateStruct", "date", default=""),
        "sponsor": _g(p, "sponsorCollaboratorsModule", "leadSponsor", "name", default=""),
        "conditions": _g(p, "conditionsModule", "conditions", default=[])[:5],
        "interventions": [
            f"{i.get('type', '')}: {i.get('name', '')}".strip(": ")
            for i in _g(p, "armsInterventionsModule", "interventions", default=[])[:5]
        ],
        "primary_outcomes": [
            f"{o.get('measure', '')} ({o.get('timeFrame', '')})"
            for o in _g(p, "outcomesModule", "primaryOutcomes", default=[])[:3]
        ],
        "brief_summary": _g(p, "descriptionModule", "briefSummary", default=""),
        "has_results": bool(study.get("hasResults")),
        "url": f"https://clinicaltrials.gov/study/{nct}",
    }


def extract_details(study: dict) -> dict:
    """Summary + eligibility + posted adverse events (only if the sponsor posted results)."""
    out = summarize_study(study)
    p = study.get("protocolSection", {})
    elig = p.get("eligibilityModule", {})
    out["eligibility"] = {
        "criteria": (elig.get("eligibilityCriteria") or "")[:3000],
        "sex": elig.get("sex", ""),
        "minimum_age": elig.get("minimumAge", ""),
        "maximum_age": elig.get("maximumAge", ""),
        "healthy_volunteers": elig.get("healthyVolunteers"),
    }
    out["secondary_outcomes"] = [
        o.get("measure", "") for o in _g(p, "outcomesModule", "secondaryOutcomes", default=[])[:5]
    ]
    ae = _g(study, "resultsSection", "adverseEventsModule", default={}) or {}
    groups = {g.get("id"): g.get("title", g.get("id")) for g in ae.get("eventGroups", [])}

    def top_events(events: list, n: int = 8) -> list[dict]:
        rows = []
        for ev in events:
            for st in ev.get("stats", []):
                at_risk = st.get("numAtRisk") or 0
                affected = st.get("numAffected") or 0
                rows.append({
                    "term": ev.get("term", ""),
                    "group": groups.get(st.get("groupId"), st.get("groupId")),
                    "affected": affected,
                    "at_risk": at_risk,
                    "pct": round(100 * affected / at_risk, 1) if at_risk else None,
                })
        rows.sort(key=lambda r: r["affected"], reverse=True)
        return rows[:n]

    out["adverse_events"] = {
        "serious": top_events(ae.get("seriousEvents", [])),
        "other": top_events(ae.get("otherEvents", [])),
        "groups": [
            {"group": g.get("title"), "serious_affected": g.get("seriousNumAffected"),
             "at_risk": g.get("seriousNumAtRisk")}
            for g in ae.get("eventGroups", [])
        ],
    } if ae else None
    return out


async def search(drug: str, condition: str | None = None, status: str | None = None,
                 phase: str | None = None, max_results: int = 5) -> dict:
    params: dict = {"query.intr": drug, "pageSize": max_results, "countTotal": "true"}
    if condition:
        params["query.cond"] = condition
    if status and status.upper() in VALID_STATUS:
        params["filter.overallStatus"] = status.upper()
    if phase:
        # e.g. "PHASE3" -> AREA[Phase]PHASE3
        params["filter.advanced"] = f"AREA[Phase]{phase.upper().replace(' ', '')}"
    data = await get(f"{BASE}/studies", params)
    studies = [summarize_study(s) for s in data.get("studies", [])]
    return {"drug": drug, "condition": condition, "total_found": data.get("totalCount"),
            "trials": studies}


async def get_study(nct_id: str) -> dict:
    data = await get(f"{BASE}/studies/{nct_id.strip().upper()}")
    return extract_details(data)
