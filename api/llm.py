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
from api.db import get_latest_summary, get_last_messages, get_all_messages, save_summary, save_suggestion_score

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


_JUDGE_SYSTEM_PROMPT = """\
你是一个专业的中文对话质量评估员。你的任务是评估一条建议回复的质量。
根据以下5个维度打分（每项1-5分），并给出简短理由。
只输出合法JSON，不要任何解释文字或Markdown。

评分维度与标准：

1. 节奏感 (rhythm)
   - 是否给对方留了回应空间？
   - 是否连续发了多条消息？
   - 是否在对方还没回应时就继续追问？
   5=节奏自然，留有空间；1=信息轰炸，让人喘不过气

2. 真诚度 (authenticity)
   - 表达是否自然，还是明显在"套话"？
   - 是否有刻意表现或过度包装的痕迹？
   - 用词是否符合两人当前关系的亲密程度？
   5=自然真实，像真人在说话；1=像客服或贺卡，明显套路

3. 推进感 (momentum)
   - 这轮对话是否推进了关系/话题深度？
   - 是否停留在表面，没有任何深入？
   - 推进是否过猛，让对方感到压力？
   5=自然推进，恰到好处；1=原地踏步或用力过猛

4. 情绪感知 (emotional_match)
   - 是否识别了对方的情绪信号？
   - 对方给出冷淡回应时，是否注意到并调整了策略？
   - 是否在对方热情时顺势推进？
   5=精准读懂情绪，回应恰当；1=对情绪信号完全忽视

5. 钩子感 (hook_quality)
   - 回复是否以自然的方式邀请对方继续聊？
   - 是否有问题、回调、小玩笑或留白让对方想回应？
   5=让人忍不住想回复；1=对话终结者

6. 人味感 (ai_naturalness) — AI味检测，越高越像真人
   - 是否出现了AI常见套路？（"我理解你的感受"、"我会支持你"、"没问题"、"加油"等空洞鼓励）
   - 回复结构是否过于工整，像在执行模板而非随口一说？
   - 是否有不必要的总结句或收尾语？（"总的来说"、"希望对你有帮助"等）
   - 语气是否足够随意、有个性，像真实的人在聊天？
   - 有没有具体细节、个人反应、或带点情绪色彩——而不是泛泛而谈？
   5=完全像真人，有个性有温度；1=明显是AI在说话，套路满满

输出格式：
{
  "rhythm": <1-5>,
  "authenticity": <1-5>,
  "momentum": <1-5>,
  "emotional_match": <1-5>,
  "hook_quality": <1-5>,
  "ai_naturalness": <1-5>,
  "justifications": {
    "rhythm": "<一句话理由>",
    "authenticity": "<一句话理由>",
    "momentum": "<一句话理由>",
    "emotional_match": "<一句话理由>",
    "hook_quality": "<一句话理由>",
    "ai_naturalness": "<一句话理由>"
  }
}"""


async def judge_reply(
    suggestion_id: int,
    conversation_id: str,
    candidate_reply: str,
) -> None:
    """
    Call GPT-4o-mini to score a candidate reply on 5 dimensions.
    Runs as a background task — never raises, logs warnings on failure.
    """
    label = {"friend": "对方", "user": "你", "llm": "你"}
    history = get_last_messages(conversation_id, n=10)
    history_text = "\n".join(
        f"{label.get(sender, sender)}: {content}" for sender, content in history
    )
    user_content = (
        f"对话记录（时间顺序）：\n{history_text}\n\n"
        f"待评估的建议回复：\n{candidate_reply}"
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.judge_model,
            "messages": messages,
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]

        # Strip markdown code fences if the model wrapped the JSON
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()

        data = json.loads(clean)
        weights = {"rhythm": 0.12, "authenticity": 0.18, "momentum": 0.15,
                   "emotional_match": 0.18, "hook_quality": 0.12, "ai_naturalness": 0.25}
        overall = sum(data.get(k, 0) * w for k, w in weights.items()) * 20
        data["overall_score"] = round(overall, 1)

        save_suggestion_score(
            suggestion_id=suggestion_id,
            conversation_id=conversation_id,
            scores=data,
            judge_model=settings.judge_model,
            raw_response=raw,
        )
        logger.info(
            "Judge score saved: suggestion_id=%s overall=%.1f", suggestion_id, overall
        )
    except Exception:
        logger.warning(
            "judge_reply failed for suggestion_id=%s", suggestion_id, exc_info=True
        )


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
