import asyncio
import json
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api.models import MessageRequest, SendUserMessageRequest, FriendMessageRequest
from api.llm import generate_replies, stream_ollama_reply, maybe_summarize, judge_reply
from api.db import (
    init_db, get_last_messages, save_message,
    log_llm_suggestion, mark_suggestion_sent,
)
from api.config import settings

_VALID_SOURCES = {"manual", "llm_accepted", "llm_modified"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Text Coach API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Text Coach API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


async def _stream_reply(conversation_id: str):
    """
    Stream Ollama reply as NDJSON chunks.
    Accumulates the full text, saves it to DB on completion, and emits a
    final {"done": true, "suggestion_id": <id>} line so the frontend gets
    the suggestion_id without a second round trip.
    Fires the judge as a background asyncio task after streaming finishes.
    """
    full_text = []
    async for chunk in stream_ollama_reply(conversation_id):
        full_text.append(chunk)
        yield json.dumps({"chunk": chunk}) + "\n"
    candidate = "".join(full_text)
    suggestion_id = log_llm_suggestion(conversation_id, candidate)
    yield json.dumps({"done": True, "suggestion_id": suggestion_id}) + "\n"
    asyncio.create_task(judge_reply(suggestion_id, conversation_id, candidate))


@app.post("/friend_message")
def friend_message(req: FriendMessageRequest):
    """Save a message received from the friend (sent=1, confirmed immediately)."""
    save_message(req.conversation_id, "friend", req.content)
    return {"status": "ok"}


@app.post("/suggest_reply")
async def suggest_reply(frd_msg: MessageRequest, background_tasks: BackgroundTasks):
    """
    Generate a reply suggestion and save it to DB as sent=0 (pending).

    Ollama  → StreamingResponse (NDJSON chunks + final {"done": true, "suggestion_id": N})
    Others  → JSON {"reply": "...", "suggestion_id": N}

    The suggestion_id is used by /send_user_message to confirm or discard the suggestion.
    After returning, a background task scores the suggestion via the LLM judge.
    """
    try:
        await maybe_summarize(frd_msg.conversation_id)
        if settings.llm_provider == "ollama":
            return StreamingResponse(
                _stream_reply(frd_msg.conversation_id),
                media_type="application/x-ndjson",
            )
        result = await generate_replies(frd_msg.conversation_id)
        reply = result.get("reply", "")
        suggestion_id = log_llm_suggestion(frd_msg.conversation_id, reply)
        background_tasks.add_task(
            judge_reply, suggestion_id, frd_msg.conversation_id, reply
        )
        return {"reply": reply, "suggestion_id": suggestion_id}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/send_user_message")
def send_user_message(req: SendUserMessageRequest):
    """
    Confirm and persist the user's chosen reply.

    source='manual'       → inserts a new sender='user' row (no LLM involved)
    source='llm_accepted' → marks the existing suggestion row sent=1
                            (suggestion_id required)
    source='llm_modified' → inserts sender='user' row; original suggestion
                            row stays sent=0 (preserves the LLM's original text)
    """
    if req.source not in _VALID_SOURCES:
        raise HTTPException(
            status_code=422,
            detail=f"source must be one of {sorted(_VALID_SOURCES)}",
        )

    if req.source == "llm_accepted":
        if req.suggestion_id is None:
            raise HTTPException(
                status_code=422,
                detail="suggestion_id is required when source='llm_accepted'",
            )
        found = mark_suggestion_sent(req.suggestion_id)
        if not found:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Suggestion {req.suggestion_id} not found, already accepted, "
                    "or belongs to a different conversation."
                ),
            )
    else:
        # 'manual' or 'llm_modified' — store as a new user row
        save_message(req.conversation_id, "user", req.content, sent=1, source=req.source)

    return {"status": "ok"}


@app.get("/get_history")
def get_history(conversation_id: str):
    """
    Return confirmed (sent=1) messages for display in the chat UI.
    Unaccepted LLM suggestions (sent=0) are excluded.
    """
    messages = get_last_messages(conversation_id, n=50)
    return {
        "messages": [{"sender": sender, "content": content} for sender, content in messages]
    }
