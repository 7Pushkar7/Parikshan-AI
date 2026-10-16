"""openFDA (free, no key needed for low volume).

  /drug/event.json  -> FAERS: real-world adverse event reports sent to the US FDA
  /drug/label.json  -> official drug label (boxed warning, warnings, adverse reactions)
Docs: https://open.fda.gov/apis/drug/
Note: FAERS counts are REPORTS, not incidence rates. A report does not prove the drug caused the event.
"""
from app.clients.http import get

BASE = "https://api.fda.gov/drug"


def _drug_query(drug: str) -> str:
    d = drug.strip().replace('"', "")
    return (f'(patient.drug.openfda.generic_name:"{d}"'
            f'+OR+patient.drug.openfda.brand_name:"{d}")')


async def _get_raw(path: str, query: str):
    # openFDA needs the "+" left unescaped, so build the query string ourselves.
    return await get(f"{BASE}/{path}?{query}", allow_404=True)


async def adverse_events(drug: str, top_n: int = 10) -> dict:
    q = _drug_query(drug)
    total = await _get_raw("event.json", f"search={q}&limit=1")
    if not total:
        return {"drug": drug, "total_reports": 0, "serious_reports": 0, "top_reactions": [],
                "note": "No FAERS reports found for this name. Try the generic name."}
    serious = await _get_raw("event.json", f"search={q}+AND+serious:1&limit=1")
    reactions = await _get_raw("event.json",
                               f"search={q}&count=patient.reaction.reactionmeddrapt.exact&limit={top_n}")
    deaths = await _get_raw("event.json", f"search={q}+AND+seriousnessdeath:1&limit=1")

    def meta_total(d):
        return (d or {}).get("meta", {}).get("results", {}).get("total", 0)

    n_total = meta_total(total)
    return {
        "drug": drug,
        "total_reports": n_total,
        "serious_reports": meta_total(serious),
        "death_reports": meta_total(deaths),
        "top_reactions": [
            {"reaction": r["term"].title(), "reports": r["count"],
             "share_pct": round(100 * r["count"] / n_total, 1) if n_total else None}
            for r in (reactions or {}).get("results", [])
        ],
        "source_url": "https://open.fda.gov/data/faers/",
        "caveat": "FAERS reports are voluntary and unverified; counts show reporting volume, not risk.",
    }


async def label(drug: str) -> dict | None:
    d = drug.strip().replace('"', "")
    data = await _get_raw("label.json",
                          f'search=(openfda.generic_name:"{d}"+OR+openfda.brand_name:"{d}")&limit=1')
    if not data or not data.get("results"):
        return None
    r = data["results"][0]

    def first(field, n=1500):
        v = r.get(field)
        return (v[0] if isinstance(v, list) and v else (v or ""))[:n]

    return {
        "brand_names": r.get("openfda", {}).get("brand_name", [])[:3],
        "generic_names": r.get("openfda", {}).get("generic_name", [])[:3],
        "boxed_warning": first("boxed_warning"),
        "warnings": first("warnings_and_cautions") or first("warnings"),
        "adverse_reactions": first("adverse_reactions"),
        "contraindications": first("contraindications", 800),
        "set_id": r.get("set_id", ""),
        "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={d}",
    }
