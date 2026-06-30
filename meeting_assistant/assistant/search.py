"""Google search via the SerpApi API (https://serpapi.com)."""

import os

import httpx

_SERPAPI_URL = "https://serpapi.com/search"


def google_search(query: str, num: int = 6, gl: str = "jp", hl: str = "ja") -> list[dict]:
    """Run a Google search through SerpApi and return simplified organic results.

    gl = geolocation country, hl = interface language. Defaults lean Japanese
    since this assists Japanese meetings, but English queries work fine too.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set")

    resp = httpx.get(
        _SERPAPI_URL,
        params={
            "engine": "google",
            "q": query,
            "num": num,
            "gl": gl,
            "hl": hl,
            "api_key": api_key,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"SerpApi error: {data['error']}")

    results = []

    # Answer box / featured snippet, when present, is high-signal — surface it first.
    if answer := data.get("answer_box"):
        snippet = answer.get("answer") or answer.get("snippet") or ""
        if snippet:
            results.append({
                "title": answer.get("title", "Answer"),
                "link": answer.get("link", ""),
                "snippet": snippet,
            })

    for item in data.get("organic_results", [])[:num]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })

    return results
