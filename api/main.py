from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from api.models import MessageRequest
from api.llm import generate_replies, stream_ollama_reply
from api.db import init_db, get_last_messages
from api.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # runs on startup
    yield      # app is running
               # anything after yield runs on shutdown


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

@app.post("/suggest_reply")
async def suggest_reply(frd_msg: MessageRequest):
    try:
        if settings.llm_provider == "ollama":
            return StreamingResponse(
                stream_ollama_reply(frd_msg.conversation_id),
                media_type="text/plain",
            )
        # openrouter / vllm — non-streaming JSON
        result = await generate_replies(frd_msg.conversation_id)
        return {"reply": result.get("reply", "")}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/get_history")
def get_history(conversation_id: str):
    messages = get_last_messages(conversation_id, n=10)  # or all messages
    # Format as list of dicts for frontend
    return {"messages": [{"sender": sender, "content": content} for sender, content in messages]}