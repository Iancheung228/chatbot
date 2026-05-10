import asyncio
import json
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api.models import MessageRequest, SendUserMessageRequest, FriendMessageRequest
from api.llm import (
    generate_reply_single, generate_reply_two_step, generate_reply_two_step_all,
    generate_reply_three_step,
    stream_single, stream_two_step, stream_two_step_all, stream_three_step,
    brainstorm_best_direction, generate_candidate_directions, select_best_direction,
    maybe_summarize, judge_reply,
)
from api.db import (
    init_db, get_last_messages, save_message,
    log_llm_suggestion, log_candidate_directions,
    mark_suggestion_sent,
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


async def _stream_reply(conversation_id: str, pipeline: str = "single", reply_model: str = "qwen-local"):
    """
    Stream Ollama reply as NDJSON chunks (qwen-local path only).
    Emits a final {"done": true, "suggestion_id": N} line.
    Fires the judge as a background asyncio task after streaming finishes.
    """
    stream_gen = None
    step1 = step2 = None

    if pipeline in ("two_step", "two_step_all"):
        try:
            brainstorm = await brainstorm_best_direction(conversation_id)
            yield json.dumps({"brainstorm": brainstorm}) + "\n"
            stream_gen = stream_two_step(conversation_id, brainstorm) if pipeline == "two_step" \
                else stream_two_step_all(conversation_id, brainstorm)
        except Exception as e:
            logger.warning("two_step brainstorm failed, falling back to single: %s", e)
            pipeline = "single"

    elif pipeline == "three_step":
        try:
            step1 = await generate_candidate_directions(conversation_id)
            yield json.dumps({"step1": step1}) + "\n"
            step2 = await select_best_direction(conversation_id, step1)
            yield json.dumps({"step2": step2}) + "\n"
            direction = {"angle": step2["angle"], "draft": step2["draft"]}
            stream_gen = stream_three_step(conversation_id, direction)
        except Exception as e:
            logger.warning("three_step analysis failed, falling back to single: %s", e)
            pipeline = "single"
            step1 = step2 = None

    if stream_gen is None:
        stream_gen = stream_single(conversation_id)

    full_text = []
    async for chunk in stream_gen:
        full_text.append(chunk)
        yield json.dumps({"chunk": chunk}) + "\n"

    candidate = "".join(full_text)
    message_id = log_llm_suggestion(
        conversation_id, candidate, pipeline=pipeline,
        reply_model=reply_model, step1=step1, step2=step2,
    )
    if pipeline == "three_step" and step1 and step2:
        log_candidate_directions(
            message_id,
            step1.get("directions", []),
            step2.get("evaluations", []),
            step2.get("selected"),
        )
    yield json.dumps({"done": True, "suggestion_id": message_id}) + "\n"
    asyncio.create_task(
        judge_reply(message_id, conversation_id, candidate,
                    pipeline=pipeline, reply_model=reply_model)
    )


@app.post("/friend_message")
def friend_message(req: FriendMessageRequest):
    """Save a message received from the friend (sent=1, confirmed immediately)."""
    save_message(req.conversation_id, "friend", req.content)
    return {"status": "ok"}


@app.post("/suggest_reply")
async def suggest_reply(frd_msg: MessageRequest, background_tasks: BackgroundTasks):
    """
    Generate a reply suggestion and save it to DB as sent=0 (pending).

    pipeline="single"       → one-shot call, full context → reply
    pipeline="two_step"     → GPT-4o-mini picks best direction → reply_model refines
    pipeline="two_step_all" → GPT-4o-mini brainstorms all → reply_model picks any
    pipeline="three_step"   → GPT-4o-mini generates 3 candidates → selects+refines → reply_model delivers

    qwen-local → StreamingResponse (NDJSON)
    Others     → JSON {"reply": "...", "suggestion_id": N}

    Judge scores every suggestion as a background task regardless of acceptance.
    """
    pipeline = frd_msg.pipeline or settings.reply_pipeline
    reply_model = frd_msg.reply_model
    try:
        await maybe_summarize(frd_msg.conversation_id)

        if reply_model == "qwen-local":
            return StreamingResponse(
                _stream_reply(frd_msg.conversation_id, pipeline=pipeline, reply_model=reply_model),
                media_type="application/x-ndjson",
            )

        # Non-streaming path
        step1 = step2 = None
        if pipeline == "two_step":
            result = await generate_reply_two_step(frd_msg.conversation_id, reply_model=reply_model)
            reply = result["reply"]
        elif pipeline == "two_step_all":
            result = await generate_reply_two_step_all(frd_msg.conversation_id, reply_model=reply_model)
            reply = result["reply"]
        elif pipeline == "three_step":
            result = await generate_reply_three_step(frd_msg.conversation_id, reply_model=reply_model)
            reply = result["reply"]
            step1 = result.get("step1")
            step2 = result.get("step2")
        else:
            result = await generate_reply_single(frd_msg.conversation_id, reply_model=reply_model)
            reply = result["reply"]

        message_id = log_llm_suggestion(
            frd_msg.conversation_id, reply, pipeline=pipeline,
            reply_model=reply_model, step1=step1, step2=step2,
        )
        if pipeline == "three_step" and step1 and step2:
            log_candidate_directions(
                message_id,
                step1.get("directions", []),
                step2.get("evaluations", []),
                step2.get("selected"),
            )

        background_tasks.add_task(
            judge_reply, message_id, frd_msg.conversation_id, reply,
            pipeline, reply_model,
        )

        response = {"reply": reply, "suggestion_id": message_id}
        if step1 is not None:
            response["step1"] = step1
        if step2 is not None:
            response["step2"] = step2
        return response
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/send_user_message")
def send_user_message(req: SendUserMessageRequest):
    """
    Confirm and persist the user's chosen reply.

    source='manual'       → inserts a new sender='user' row
    source='llm_accepted' → marks the existing suggestion row sent=1
    source='llm_modified' → inserts sender='user' row; original suggestion stays sent=0
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
        save_message(req.conversation_id, "user", req.content, sent=1, source=req.source)
    return {"status": "ok"}


@app.get("/get_history")
def get_history(conversation_id: str):
    """Return confirmed (sent=1) messages for display in the chat UI."""
    messages = get_last_messages(conversation_id, n=50)
    return {
        "messages": [{"sender": sender, "content": content} for sender, content in messages]
    }