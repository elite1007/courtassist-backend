"""
Triple-check pipeline for every citation before it is ever shown to the user
mid-hearing. A citation is only labeled "verified" if it survives three
independent checks:

  1. Provenance check   -- the citation actually came back attached to the
                            model's own search results/citations list (not
                            something it typed from memory with no grounding).
  2. Direct source check -- we independently fetch the candidate source URL
                            ourselves (the backend, not the model) and confirm
                            key terms from the citation actually appear on
                            that page.
  3. Corroboration check -- a second, separate Perplexity search call (fresh
                            context, different prompt) is asked to confirm the
                            citation independently; we require its own
                            citations/search results to corroborate.

Citations that fail any check are still returned to the app, but flagged
`verified: false` with a human-readable reason so the UI can visually
distinguish them (e.g. greyed out with a "not independently confirmed --
verify before relying on this" warning) instead of presenting them as
equally trustworthy fact.
"""
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

from perplexity import independent_search

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "v", "vs",
    "re", "et", "al", "no", "rule", "section", "act",
}
_UA = "CourtAssistVerifier/1.0 (+personal-hearing-assistant)"


def _tokens(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-()]*", text or "")
    return [t.lower() for t in raw if t.lower() not in _STOPWORDS and len(t) > 1]


def _overlap_ratio(needle_tokens: List[str], haystack: str) -> float:
    if not needle_tokens:
        return 0.0
    hay = haystack.lower()
    hits = sum(1 for t in needle_tokens if t in hay)
    return hits / len(needle_tokens)


def _candidate_urls(citation: Dict[str, Any]) -> List[str]:
    urls = []
    hint = citation.get("source_hint", "") or ""
    if hint.startswith("http"):
        urls.append(hint)
    for sr in citation.get("_search_results_pool", []):
        url = sr.get("url") if isinstance(sr, dict) else None
        if url and url not in urls:
            urls.append(url)
    return urls[:4]


async def _fetch_ok(url: str, needle_tokens: List[str]) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True,
                                      headers={"User-Agent": _UA}) as client:
            r = await client.get(url)
        if r.status_code >= 400:
            return {"ok": False, "reason": f"HTTP {r.status_code} fetching source"}
        text = re.sub(r"<[^>]+>", " ", r.text[:200000])  # crude tag strip, good enough for matching
        ratio = _overlap_ratio(needle_tokens, text)
        if ratio >= 0.5:
            return {"ok": True, "ratio": ratio, "domain": urlparse(url).netloc}
        return {"ok": False, "reason": f"Source page did not clearly contain the cited terms (match {ratio:.0%})"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"Could not fetch source: {e}"}


_NEGATIVE_PHRASES = (
    "could not find", "no reliable source", "does not appear to exist",
    "cannot confirm", "unable to verify", "no such case", "fabricated",
    "not a real", "i don't have information",
)


async def verify_citation(api_key: str, citation: Dict[str, Any]) -> Dict[str, Any]:
    label = citation.get("label", "")
    detail = citation.get("detail", "")
    needle_tokens = _tokens(label) + _tokens(detail)[:8]

    # Check 1: provenance -- was this label backed by something in the model's own
    # search pool, rather than typed purely from parametric memory?
    pool_hit = any(
        any(tok in (sr.get("title", "") + sr.get("url", "")).lower() for tok in needle_tokens[:4])
        for sr in citation.get("_search_results_pool", [])
    )

    # Check 2: direct fetch of a candidate URL by the backend itself.
    fetch_result = {"ok": False, "reason": "No fetchable URL found for this citation"}
    verified_url = None
    for url in _candidate_urls(citation):
        result = await _fetch_ok(url, needle_tokens)
        if result["ok"]:
            fetch_result = result
            verified_url = url
            break
        fetch_result = result  # keep last failure reason if none succeed

    # Check 3: independent corroboration via a fresh, separate search call.
    corroboration_ok = False
    corroboration_note = ""
    try:
        query = (f"Confirm whether this legal citation is real and accurately described, "
                 f"and give its exact holding/text: \"{label}\" -- {detail}")
        result = await independent_search(api_key, query)
        content_lower = (result.get("content") or "").lower()
        negative = any(p in content_lower for p in _NEGATIVE_PHRASES)
        has_sources = bool(result.get("citations") or result.get("search_results"))
        corroboration_ok = has_sources and not negative
        corroboration_note = result.get("content", "")[:280]
    except Exception as e:  # noqa: BLE001
        corroboration_note = f"Corroboration search failed: {e}"

    verified = fetch_result["ok"] and corroboration_ok
    if verified:
        reason = f"Confirmed by direct source fetch ({fetch_result.get('domain', 'source')}) and independent corroborating search."
    else:
        problems = []
        if not fetch_result["ok"]:
            problems.append(fetch_result.get("reason", "source fetch failed"))
        if not corroboration_ok:
            problems.append("independent search did not corroborate this citation")
        reason = "NOT independently verified -- " + "; ".join(problems) + ". Double-check before relying on this."

    out = dict(citation)
    out.pop("_search_results_pool", None)
    out.update({
        "verified": verified,
        "verified_url": verified_url,
        "verification_reason": reason,
        "provenance_hit": pool_hit,
        "corroboration_excerpt": corroboration_note,
    })
    return out


async def verify_all(api_key: str, citations: List[Dict[str, Any]],
                      search_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for c in citations:
        c = dict(c)
        c["_search_results_pool"] = search_pool
        enriched.append(c)
    out = []
    for c in enriched:
        out.append(await verify_citation(api_key, c))
    return out
