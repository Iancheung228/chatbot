"""
LLM backend — single entry point for all providers.

Switch providers by setting LLM_PROVIDER in .env:
  openrouter  — cloud API via OpenRouter (default, dev)
  ollama      — local Ollama server with GGUF model (local fine-tuned testing)
  vllm        — cloud vLLM server (production, fine-tuned model)
"""
import httpx
import json
import logging
from api.config import settings
from api.db import get_latest_summary, get_last_messages, get_all_messages, save_summary

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT: str = ""

def _get_system_prompt() -> str:
    """Load system prompt once and cache it for the lifetime of the process."""
    global _SYSTEM_PROMPT
    if not _SYSTEM_PROMPT:
        with open(settings.system_prompt_file, "r", encoding="utf-8") as f:
            _SYSTEM_PROMPT = f.read().strip()
    return _SYSTEM_PROMPT


def build_system_prompt(conversation_id: str) -> str:
    """System message: persona instructions with {SUMMARY} injected."""
    base = _get_system_prompt()
    summary = get_latest_summary(conversation_id)
    summary_block = (
        f"Prior conversation summary:\n{summary}"
        if summary
        else "(none — conversation is short enough to fit in full below)"
    )
    return base.replace("{SUMMARY}", summary_block)


def build_api_payload(conversation_id: str) -> list[dict]:
    """
    Build the messages array for the LLM call.

    History is passed as proper ChatML turns — the same format the model
    was trained on. This prevents the model from pattern-matching the flat
    text history against the examples and hallucinating a continuation.

        DB sender "friend" → role "user"      (the person we reply to)
        DB sender "user"   → role "assistant" (our previous replies)
        DB sender "llm"    → role "assistant" (accepted LLM suggestions)

    Consecutive messages from the same role are merged (WeChat short bursts).
    The friend's latest message is already the last DB entry (saved before
    /suggest_reply is called), so the model sees it as the open user turn
    it needs to respond to.
    """
    role_map = {"friend": "user", "user": "assistant", "llm": "assistant"}
    history = get_last_messages(conversation_id, n=20)

    chat_messages: list[dict] = []
    for sender, content in history:
        role = role_map.get(sender, "user")
        if chat_messages and chat_messages[-1]["role"] == role:
            chat_messages[-1]["content"] += "\n" + content
        else:
            chat_messages.append({"role": role, "content": content})

    messages = [
        {"role": "system", "content": build_system_prompt(conversation_id)},
        *chat_messages,
    ]

    logger.info(
        "\n" + "=" * 60 + "\n"
        "LLM PAYLOAD  conversation_id=%s\n"
        + "=" * 60 + "\n"
        + json.dumps(messages, ensure_ascii=False, indent=2)
        + "\n" + "=" * 60,
        conversation_id,
    )

    return messages


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
        "messages": build_api_payload(conversation_id),
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
        "messages": build_api_payload(conversation_id),
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
        "messages": build_api_payload(conversation_id),
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


_HISTORY_WINDOW = 20   # messages kept verbatim in every LLM call
_SUMMARIZE_EVERY = 5   # summarize once per N new messages beyond the window


async def _generate_summary(overflow: list[tuple[str, str]], existing_summary: str) -> str:
    """
    Call the LLM to produce an updated summary of overflow messages.
    Uses OpenRouter/vLLM if available, falls back to Ollama.
    Always uses a cheap/fast model — this is a housekeeping call, not a reply.
    """
    label = {"friend": "Friend", "user": "You", "llm": "You"}
    transcript = "\n".join(
        f"{label.get(sender, sender)}: {content}" for sender, content in overflow
    )
    prior_block = f"Existing summary:\n{existing_summary}\n\n" if existing_summary else ""
    prompt = (
        f"{prior_block}"
        f"New messages to incorporate:\n{transcript}\n\n"
        "Write a concise factual summary (3-6 sentences) of the relationship context, "
        "key topics discussed, and emotional tone so far. "
        "This summary will be shown to an AI playing the role of a close friend "
        "so it can maintain continuity. Do not invent anything not in the messages."
    )
    messages = [{"role": "user", "content": prompt}]

    provider = settings.llm_provider
    if provider == "ollama":
        payload = {"model": settings.ollama_model, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    if provider == "vllm":
        base_url, api_key, model = settings.vllm_base_url, settings.vllm_api_key, settings.vllm_model
    else:
        base_url, api_key, model = settings.base_url, settings.openrouter_api_key, settings.model_name

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={"model": model, "messages": messages, "temperature": 0.3},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def maybe_summarize(conversation_id: str) -> None:
    """
    Trigger a summary update if the conversation has grown beyond the history window.
    Runs every _SUMMARIZE_EVERY messages past the window so the summary stays fresh
    without making an extra LLM call on every single turn.
    """
    all_messages = get_all_messages(conversation_id)
    total = len(all_messages)

    if total <= _HISTORY_WINDOW:
        return  # everything fits in the window — no summary needed

    overflow_count = total - _HISTORY_WINDOW
    if overflow_count % _SUMMARIZE_EVERY != 0:
        return  # not a summarization checkpoint yet

    overflow = all_messages[:overflow_count]
    existing_summary = get_latest_summary(conversation_id)

    logger.info(
        "Summarizing conversation_id=%s: %d overflow messages (existing summary: %s)",
        conversation_id,
        len(overflow),
        "yes" if existing_summary else "none",
    )

    summary = await _generate_summary(overflow, existing_summary)
    save_summary(conversation_id, summary)
    logger.info("Summary saved for conversation_id=%s", conversation_id)


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
