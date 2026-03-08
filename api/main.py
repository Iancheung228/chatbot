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
async def suggest_reply(frd_msg: MessageRequest):

    #### Instead of saving friend's msg here, we save it in frontend.py
    from app.db import init_db, save_message
    # # Save the user message to the database
    # save_message(frd_msg.conversation_id, "friend", frd_msg.message)

    # generate LLM reply
    result = await generate_replies(frd_msg.message, frd_msg.conversation_id)

    # Save only the first LLM suggestion to the database
    suggestions = result.get("suggestions", [])
    # I do not want to save bot's first reply automatically, i want the user to choose their prefered msg.
    #if suggestions:
    #    save_message(frd_msg.conversation_id, "bot", suggestions[0].get("reply", ""))

    return {
        "input": frd_msg.message,
        "motivation_analysis": result.get("motivation_analysis"),
        "suggestions": result.get("suggestions")
    }

from app.db import get_last_messages

@app.get("/get_history")
def get_history(conversation_id: str):
    messages = get_last_messages(conversation_id, n=10)  # or all messages
    # Format as list of dicts for frontend
    return {"messages": [{"sender": sender, "content": content} for sender, content in messages]}