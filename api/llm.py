"""
LLM backend — single entry point for all providers.

Switch providers by setting LLM_PROVIDER in .env:
  openrouter  — cloud API via OpenRouter (default, dev)
  ollama      — local Ollama server with GGUF model (local fine-tuned testing)
  vllm        — cloud vLLM server (production, fine-tuned model)
"""
import httpx
import json
from api.config import settings
from api.db import get_latest_summary, get_last_messages


_SYSTEM_PROMPT: str = ""

def _get_system_prompt() -> str:
    """Load system prompt once and cache it for the lifetime of the process."""
    global _SYSTEM_PROMPT
    if not _SYSTEM_PROMPT:
        with open(settings.system_prompt_file, "r", encoding="utf-8") as f:
            _SYSTEM_PROMPT = f.read().strip()
    return _SYSTEM_PROMPT


def build_system_prompt(conversation_id: str) -> str:
    """System message: persona instructions + optional prior summary. No history."""
    parts = [_get_system_prompt()]
    summary = get_latest_summary(conversation_id)
    if summary:
        parts.append(f"---\n\nConversation summary so far:\n{summary}")
    return "\n\n".join(parts)


def build_messages(conversation_id: str) -> list[dict]:
    """
    Build the full messages array for the LLM call.

    Conversation history from DB is mapped to ChatML roles:
        DB sender "friend"  → role "user"      (the person we reply to)
        DB sender "user"    → role "assistant"  (our previous replies)

    The current friend message is already the last DB entry — it was saved
    in frontend.py before /suggest_reply is called.
    Consecutive messages from the same sender are merged (WeChat short bursts).
    """
    role_map = {"friend": "user", "user": "assistant"}
    history = get_last_messages(conversation_id, n=20)

    chat_messages: list[dict] = []
    for sender, content in history:
        role = role_map.get(sender, "user")
        if chat_messages and chat_messages[-1]["role"] == role:
            chat_messages[-1]["content"] += "\n" + content
        else:
            chat_messages.append({"role": role, "content": content})

    return [
        {"role": "system", "content": build_system_prompt(conversation_id)},
        *chat_messages,
    ]


async def _call_openai_compatible(
    conversation_id: str,
    base_url: str,
    api_key: str,
    model: str,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": build_messages(conversation_id),
        "temperature": 0.8,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected LLM response shape: {data}") from e
    return {"reply": content}


async def _call_ollama(conversation_id: str) -> dict:
    payload = {
        "model": settings.ollama_model,
        "messages": build_messages(conversation_id),
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
    return {"reply": content}


async def stream_ollama_reply(conversation_id: str):
    """Async generator: yields text chunks from Ollama as they're produced."""
    payload = {
        "model": settings.ollama_model,
        "messages": build_messages(conversation_id),
        "stream": True,
        "options": {"num_predict": 400},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{settings.ollama_base_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done"):
                        yield data["message"]["content"]


async def generate_replies(conversation_id: str) -> dict:
    """
    Generate a reply for the given user message.
    Returns {"reply": "<natural language reply>"}.
    Provider is selected via LLM_PROVIDER in .env.
    """
    provider = settings.llm_provider

    if provider == "ollama":
        return await _call_ollama(conversation_id)

    if provider == "vllm":
        return await _call_openai_compatible(
            conversation_id,
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_model,
        )

    # Default: openrouter
    return await _call_openai_compatible(
        conversation_id,
        base_url=settings.base_url,
        api_key=settings.openrouter_api_key,
        model=settings.model_name,
    )


# def ocr_to_smart_text(ocr_text: str, prompt: str = None, model: str = None) -> str:
#     """Synchronous LLM call for OCR post-processing (data pipeline only)."""
#     ocr_correction_prompt = prompt or (
#         "You are given a conversation between 2 friends in Simplified Chinese. "
#         "The conversation transcript was obtained via OCR and could contain errors. "
#         "Correct potential errors by leveraging context. Prioritize one-character corrections "
#         "and visually plausible replacements over contextual reinterpretations. "
#         "Keep format the same as input."
#     )
#     payload = {
#         "model": model or settings.model_name,
#         "messages": [
#             {"role": "system", "content": ocr_correction_prompt},
#             {"role": "user", "content": ocr_text},
#         ],
#         "temperature": 0.3,
#     }
#     headers = {
#         "Authorization": f"Bearer {settings.openrouter_api_key}",
#         "Content-Type": "application/json",
#     }
#     response = httpx.post(
#         f"{settings.base_url}/chat/completions",
#         headers=headers,
#         json=payload,
#     )
#     response.raise_for_status()
#     return response.json()["choices"][0]["message"]["content"]
