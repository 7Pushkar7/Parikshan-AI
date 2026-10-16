"""Checks that the REAL services are reachable before you run or demo the agent.

    python scripts/check_setup.py

It makes 1 real Gemini call and 1 small request to each medical API, and prints OK or the problem.
"""
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
ok_all = True


def check(name, fn):
    global ok_all
    t0 = time.perf_counter()
    try:
        detail = fn()
        print(f"  OK    {name:<28} {detail}  ({(time.perf_counter() - t0):.1f}s)")
    except Exception as exc:  # noqa: BLE001
        ok_all = False
        print(f"  FAIL  {name:<28} {type(exc).__name__}: {str(exc)[:200]}")


def gemini_call(model: str) -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is empty. Put your key in the .env file.")
    from google import genai
    client = genai.Client(api_key=key)  # keep a reference: the SDK closes an unreferenced client
    for attempt in range(3):  # Google sometimes answers 503 "high demand" for a few seconds
        try:
            resp = client.models.generate_content(model=model, contents="Reply with the word: ready")
            break
        except Exception as exc:  # noqa: BLE001
            if "503" in str(exc) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    u = resp.usage_metadata
    return f"{model} replied '{resp.text.strip()[:20]}' ({u.prompt_token_count} in / {u.candidates_token_count} out tokens)"


def gemini():
    return gemini_call(os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))


def gemini_backup():
    return gemini_call(os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite"))


def pubmed():
    r = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                  params={"db": "pubmed", "term": "semaglutide", "retmode": "json", "retmax": 1}, timeout=20)
    r.raise_for_status()
    return f"{int(r.json()['esearchresult']['count']):,} papers found for semaglutide"


def clinical_trials():
    r = httpx.get("https://clinicaltrials.gov/api/v2/studies",
                  params={"query.intr": "semaglutide", "pageSize": 1, "countTotal": "true"}, timeout=20)
    r.raise_for_status()
    return f"{r.json().get('totalCount', 0):,} trials registered for semaglutide"


def openfda():
    r = httpx.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:"semaglutide"&limit=1',
                  timeout=20)
    r.raise_for_status()
    return f"{r.json()['meta']['results']['total']:,} FAERS reports for semaglutide"


print("\nChecking real services...\n")
check("Gemini main model", gemini)
check("Gemini backup model", gemini_backup)
check("PubMed (NCBI E-utilities)", pubmed)
check("ClinicalTrials.gov API v2", clinical_trials)
check("openFDA (FAERS)", openfda)
if ok_all:
    print("\nAll good. Start the app with:  uvicorn app.main:app --reload\n")
else:
    print("\nSome checks failed. If only ONE Gemini line failed with 503 or 429, you can still run the app:"
          "\nit switches to the other model automatically. 503 = Google busy, try again in a few minutes."
          "\n429 = free-tier limit, wait a minute. 400/401/403 = check GEMINI_API_KEY in .env.\n")
sys.exit(0 if ok_all else 1)
