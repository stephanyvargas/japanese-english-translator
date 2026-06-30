"""FastAPI backend for the meeting assistant.

  POST /assistant  — ask a question; Claude answers, searching Google (Serper) as needed
  GET  /health     — Cloud Run health probe
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("assistant")

from assistant.agent import run as run_agent  # noqa: E402
from assistant.models import AssistantReply, AssistantRequest  # noqa: E402

app = FastAPI(title="Meeting Assistant API")

# Comma-separated origins via ALLOWED_ORIGINS env; falls back to "*" for local dev.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
_allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/assistant", response_model=AssistantReply)
async def assistant(req: AssistantRequest):
    client = anthropic.Anthropic()
    loop = asyncio.get_event_loop()
    log.info("ask model=%s ctx=%dch msg=%r", req.model, len(req.context), req.message[:80])

    reply = await loop.run_in_executor(
        _executor,
        lambda: run_agent(
            req.message, client,
            context=req.context, model=req.model, history=req.history,
        ),
    )
    log.info("done searches=%d sources=%d", len(reply.searched), len(reply.sources))
    return reply


# ── Static frontend (dev convenience) ────────────────────────────────────────

_frontend = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
