"""Shared async HTTP helper with simple retry for the external biomedical APIs."""
import asyncio
import logging

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class ExternalAPIError(RuntimeError):
    pass


async def get(url: str, params: dict | None = None, *, expect: str = "json", retries: int = 3,
              allow_404: bool = False):
    """GET a URL and return parsed JSON (expect="json") or text (expect="text").

    Retries on rate limits / server errors with exponential backoff (1s, 2s, 4s).
    If allow_404 is True, a 404 returns None (openFDA uses 404 for "no matches").
    """
    settings = get_settings()
    headers = {"User-Agent": f"ParikshanAI/1.0 ({settings.ncbi_email})"}
    delay = 1.0
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=settings.http_timeout, headers=headers,
                                 follow_redirects=True) as client:
        for attempt in range(retries + 1):
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == 404 and allow_404:
                    return None
                if resp.status_code in RETRY_STATUS and attempt < retries:
                    log.warning("HTTP %s from %s, retrying in %.0fs", resp.status_code, url, delay)
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                return resp.json() if expect == "json" else resp.text
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < retries and not isinstance(exc, httpx.HTTPStatusError):
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                break
    raise ExternalAPIError(f"Request to {url} failed: {last_exc}")
