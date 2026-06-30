from pydantic import BaseModel


class Turn(BaseModel):
    role: str   # "user" or "assistant"
    text: str


class AssistantRequest(BaseModel):
    message: str
    context: str = ""           # pasted meeting context / transcript (transcript-ready hook)
    model: str = "sonnet"
    history: list[Turn] = []    # prior turns for multi-turn conversation


class Source(BaseModel):
    title: str
    link: str
    snippet: str


class AssistantReply(BaseModel):
    reply: str
    sources: list[Source] = []
    searched: list[str] = []    # the Google queries that were run
