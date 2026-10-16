"""PubMed via NCBI E-Utilities (free, no key needed).

Two calls:
  1. esearch  -> list of PubMed IDs (PMIDs) for a search term
  2. efetch   -> full records (title, abstract, journal, year) for those PMIDs as XML
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""
import asyncio
import xml.etree.ElementTree as ET

from app.clients.http import get
from app.config import get_settings

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_lock = asyncio.Semaphore(2)  # stay under NCBI's 3 requests/second limit


def _common_params() -> dict:
    s = get_settings()
    p = {"tool": "parikshan-ai", "email": s.ncbi_email}
    if s.ncbi_api_key:
        p["api_key"] = s.ncbi_api_key
    return p


async def search_pmids(term: str, max_results: int = 5, sort: str = "relevance") -> tuple[int, list[str]]:
    params = {**_common_params(), "db": "pubmed", "term": term, "retmode": "json",
              "retmax": max_results, "sort": sort}
    async with _lock:
        data = await get(f"{BASE}/esearch.fcgi", params)
        await asyncio.sleep(0.35)
    res = data.get("esearchresult", {})
    return int(res.get("count", 0)), res.get("idlist", [])


def parse_efetch_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    articles = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID", default="")
        a = art.find(".//MedlineCitation/Article")
        if a is None:
            continue
        title = "".join(a.find("ArticleTitle").itertext()).strip() if a.find("ArticleTitle") is not None else ""
        parts = []
        for ab in a.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = "".join(ab.itertext()).strip()
            parts.append(f"{label}: {text}" if label else text)
        journal = a.findtext(".//Journal/Title", default="")
        year = (a.findtext(".//Journal/JournalIssue/PubDate/Year")
                or (a.findtext(".//Journal/JournalIssue/PubDate/MedlineDate") or "")[:4])
        pub_types = [pt.text for pt in a.findall(".//PublicationTypeList/PublicationType") if pt.text]
        authors = []
        for au in a.findall(".//AuthorList/Author")[:3]:
            last, init = au.findtext("LastName"), au.findtext("Initials")
            if last:
                authors.append(f"{last} {init or ''}".strip())
        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": "\n".join(parts),
            "journal": journal,
            "year": year,
            "publication_types": pub_types,
            "authors": authors,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return articles


async def fetch_articles(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    params = {**_common_params(), "db": "pubmed", "id": ",".join(pmids),
              "rettype": "abstract", "retmode": "xml"}
    async with _lock:
        xml_text = await get(f"{BASE}/efetch.fcgi", params, expect="text")
        await asyncio.sleep(0.35)
    return parse_efetch_xml(xml_text)


async def search(term: str, max_results: int = 5) -> dict:
    total, pmids = await search_pmids(term, max_results)
    articles = await fetch_articles(pmids)
    return {"query": term, "total_found": total, "articles": articles}
