"""Turn uploaded documents and GitHub repos into profile text for the copilot.

Two converters, both returning plain text the client stores on the profile doc:
- extract_document: PDF → one Haiku call with a native document block (handles
  scanned PDFs, no extraction dependencies); txt/md → decoded directly, no
  model call at all.
- summarize_repo: GitHub public API (metadata + README) → one Haiku call →
  a compact project blurb the hint prompt can ground bullets in.
"""

from __future__ import annotations

import base64
import re

import anthropic
import requests

_MODEL = "claude-haiku-4-5"
_DOC_CHAR_CAP = 15000      # per document, keeps the compiled profile promptable
_SUMMARY_CHAR_CAP = 2000


def _text_of(msg) -> str:
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def extract_document(filename: str, data: bytes, client: anthropic.Anthropic) -> str:
    """Return the document's content as clean text, capped for prompting."""
    lower = filename.lower()
    if lower.endswith((".txt", ".md", ".markdown")):
        return data.decode("utf-8", errors="replace")[:_DOC_CHAR_CAP]

    if not lower.endswith(".pdf"):
        raise ValueError("Only PDF, .txt, and .md files are supported.")

    with client.messages.stream(
        model=_MODEL,
        max_tokens=8000,
        system=("Extract this document's content as clean markdown for use as an "
                "interview-preparation profile. Keep every concrete fact — names, "
                "employers, dates, technologies, numbers, achievements. Drop layout "
                "artifacts. No commentary, just the content. At most 4000 words."),
        messages=[{
            "role": "user",
            "content": [{
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.standard_b64encode(data).decode(),
                },
            }],
        }],
    ) as stream:
        msg = stream.get_final_message()
    text = _text_of(msg)
    if not text:
        raise ValueError("Could not extract any text from this PDF.")
    return text[:_DOC_CHAR_CAP]


_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def summarize_repo(repo: str, client: anthropic.Anthropic) -> str:
    """Fetch a public GitHub repo's metadata + README and blurb it for the profile."""
    repo = repo.strip().removeprefix("https://github.com/").strip("/")
    if not _REPO_RE.match(repo):
        raise ValueError('Repository must look like "owner/name".')

    meta_r = requests.get(f"https://api.github.com/repos/{repo}", timeout=15,
                          headers={"Accept": "application/vnd.github+json"})
    if meta_r.status_code == 404:
        raise ValueError(f"GitHub repo {repo} not found (private repos aren't supported).")
    if meta_r.status_code == 403:
        raise ValueError("GitHub API rate limit hit — try again in a few minutes.")
    meta_r.raise_for_status()
    meta = meta_r.json()

    readme = ""
    readme_r = requests.get(f"https://api.github.com/repos/{repo}/readme", timeout=15,
                            headers={"Accept": "application/vnd.github+json"})
    if readme_r.ok:
        readme = base64.b64decode(readme_r.json().get("content", "")) \
            .decode("utf-8", errors="replace")[:20000]

    facts = (f"Repository: {repo}\n"
             f"Description: {meta.get('description') or '—'}\n"
             f"Language: {meta.get('language') or '—'} · "
             f"Topics: {', '.join(meta.get('topics', [])) or '—'} · "
             f"Stars: {meta.get('stargazers_count', 0)}\n\n"
             f"README:\n{readme or '(no README)'}")

    with client.messages.stream(
        model=_MODEL,
        max_tokens=1024,
        system=("Summarize this GitHub project for the author's interview-prep "
                "profile: what it does, the tech stack, and the notable engineering "
                "decisions or results — the things worth mentioning in an interview. "
                "≤10 short lines, plain text."),
        messages=[{"role": "user", "content": facts}],
    ) as stream:
        msg = stream.get_final_message()
    summary = _text_of(msg)
    if not summary:
        raise ValueError("Could not summarize this repository.")
    return summary[:_SUMMARY_CHAR_CAP]
