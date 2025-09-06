from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str
    conversation_id: str
