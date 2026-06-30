"""Google search via the Serper API (https://serper.dev)."""

import os

import httpx

_SERPER_URL = "https://google.serper.dev/search"


def google_search(query: str, num: int = 6, gl: str = "jp", hl: str = "ja") -> list[dict]:
    """Run a Google search through Serper and return simplified organic results.

    gl = geolocation country, hl = interface language. Defaults lean Japanese
    since this assists Japanese meetings, but English queries work fine too.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set")

    resp = httpx.post(
        _SERPER_URL,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": num, "gl": gl, "hl": hl},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("organic", [])[:num]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })

    # Serper's answer box / knowledge graph, when present, is high-signal.
    if answer := data.get("answerBox"):
        snippet = answer.get("answer") or answer.get("snippet") or ""
        if snippet:
            results.insert(0, {
                "title": answer.get("title", "Answer box"),
                "link": answer.get("link", ""),
                "snippet": snippet,
            })

    return results
