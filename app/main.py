from fastapi import FastAPI
from app.models import MessageRequest
from app.llm import generate_replies
from app.db import init_db  # <-- import your init_db function


app = FastAPI(title="Text Coach API")

@app.on_event("startup")
def startup_event():
    init_db()  # <-- create tables at app startup


@app.get("/")
def root():
    return {"message": "Text Coach API is running!"}

@app.post("/suggest_reply")
async def suggest_reply(req: MessageRequest):
    # generate LLM reply
    result = await generate_replies(req.message, req.conversation_id)


    from app.db import init_db, save_message
    # Save the user message to the database
    save_message(req.conversation_id, "user", req.message)
    # Save only the first LLM suggestion to the database
    suggestions = result.get("suggestions", [])
    if suggestions:
        save_message(req.conversation_id, "bot", suggestions[0].get("reply", ""))

    return {
        "input": req.message,
        "motivation_analysis": result.get("motivation_analysis"),
        "suggestions": result.get("suggestions")
    }