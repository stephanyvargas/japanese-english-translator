"""Claude tool-use loop backed by Serper Google search."""

import anthropic

from .models import AssistantReply, Source, Turn
from .prompts import SYSTEM_PROMPT
from .search import google_search

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
}

_SEARCH_TOOL = {
    "name": "google_search",
    "description": (
        "Search Google for current, factual, or company-specific information. "
        "Use this whenever you are not certain or the user asks about something "
        "current. Returns a list of organic results with title, link, and snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "num": {"type": "integer", "description": "How many results (default 6)."},
        },
        "required": ["query"],
    },
}

_MAX_TURNS = 6  # safety cap on the tool-use loop


def run(
    message: str,
    client: anthropic.Anthropic,
    context: str = "",
    model: str = "sonnet",
    history: list[Turn] | None = None,
) -> AssistantReply:
    """Run one assistant turn, letting Claude search Google as needed."""
    model_id = MODEL_ALIASES.get(model.lower(), model)

    user_content = message
    if context.strip():
        user_content = f"Meeting context / transcript:\n{context.strip()}\n\n---\n\n{message}"

    messages: list[dict] = []
    for turn in (history or []):
        messages.append({"role": turn.role, "content": turn.text})
    messages.append({"role": "user", "content": user_content})

    sources: list[dict] = []
    searched: list[str] = []
    seen_links: set[str] = set()

    for _ in range(_MAX_TURNS):
        with client.messages.stream(
            model=model_id,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[_SEARCH_TOOL],
            messages=messages,
        ) as stream:
            msg = stream.get_final_message()

        if msg.stop_reason != "tool_use":
            reply_text = next((b.text for b in msg.content if b.type == "text"), "")
            return AssistantReply(
                reply=reply_text.strip(),
                sources=[Source(**s) for s in sources],
                searched=searched,
            )

        # Echo the assistant's tool-call turn back, then answer every tool_use block.
        messages.append({"role": "assistant", "content": msg.content})
        tool_results = []
        for block in msg.content:
            if block.type != "tool_use":
                continue
            query = block.input.get("query", "")
            num = int(block.input.get("num", 6) or 6)
            searched.append(query)
            try:
                results = google_search(query, num=num)
                for r in results:
                    if r["link"] and r["link"] not in seen_links:
                        seen_links.add(r["link"])
                        sources.append(r)
                content = _format_results(results) if results else "No results found."
            except Exception as exc:
                content = f"Search failed: {exc}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })
        messages.append({"role": "user", "content": tool_results})

    # Hit the loop cap — return whatever text the model last produced.
    reply_text = next((b.text for b in msg.content if b.type == "text"), "")
    return AssistantReply(
        reply=(reply_text or "I gathered some results but ran out of search steps.").strip(),
        sources=[Source(**s) for s in sources],
        searched=searched,
    )


def _format_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n{r['link']}\n{r['snippet']}")
    return "\n\n".join(lines)
