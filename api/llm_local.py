"""
Ollama client for serving a locally fine-tuned model.
Exposes the same interface as llm.py so the backend can swap between
OpenRouter and Ollama by changing LLM_BACKEND in .env.
"""
import httpx
import json
from api.config import settings
from api.llm import build_prompt


async def generate_replies(user_message: str, conversation_id: str):
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": build_prompt(conversation_id, user_message)},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    content = data["message"]["content"]
    try:
        # Strip markdown code fences if model wraps JSON in ```json ... ```
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except json.JSONDecodeError:
        return content
