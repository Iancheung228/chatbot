from pydantic import BaseModel, Field
from typing import Optional


class MessageRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    pipeline: str = "single"        # "single" | "two_step" | "two_step_all" | "three_step"
    reply_model: str = "qwen-local" # key into REPLY_MODELS registry in llm.py


class FriendMessageRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=4000)


class SendUserMessageRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=4000)
    # 'manual'       — user typed from scratch
    # 'llm_accepted' — user accepted an LLM suggestion exactly (suggestion_id required)
    # 'llm_modified' — user started from a suggestion but edited it
    source: str
    suggestion_id: Optional[int] = None
