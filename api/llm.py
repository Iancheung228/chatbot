"""
LLM backend — single entry point for all providers.

Switch providers by setting LLM_PROVIDER in .env:
  openrouter  — cloud API via OpenRouter (default, dev)
  ollama      — local Ollama server with GGUF model (local fine-tuned testing)
  vllm        — cloud vLLM server (production, fine-tuned model)

Switch pipelines by setting REPLY_PIPELINE in .env:
  single       — one-shot: full context → reply
  two_step     — GPT-4o-mini picks best direction → reply_model refines
  two_step_all — GPT-4o-mini brainstorms all directions → reply_model picks any
  three_step   — GPT-4o-mini generates 3 candidates → GPT-4o-mini selects+refines → reply_model delivers
"""
import httpx
import json
import logging
from api.config import settings
from api.db import get_latest_summary, get_last_messages, get_all_messages, save_summary, save_judge_score

logger = logging.getLogger(__name__)


# ── Prompt loader ─────────────────────────────────────────────────────────────

_prompt_cache: dict[str, str] = {}


def _load_prompt(path: str) -> str:
    """Load a prompt file once and cache by path. Restart server to refresh."""
    if path not in _prompt_cache:
        with open(path, "r", encoding="utf-8") as f:
            _prompt_cache[path] = f.read().strip()
        logger.info("Loaded prompt: %s", path)
    return _prompt_cache[path]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_chat_turns(conversation_id: str, n: int = 20) -> list[dict]:
    """
    Build ChatML turns from DB history.

    DB sender mapping:
      "friend" → role "user"      (the person we reply to)
      "user"   → role "assistant" (our previous replies)
      "llm"    → role "assistant" (accepted LLM suggestions)

    Consecutive same-role messages are merged (WeChat burst behaviour).
    The friend's latest message is the last DB entry (saved before /suggest_reply),
    so it arrives as an open "user" turn the model needs to respond to.
    """
    role_map = {"friend": "user", "user": "assistant", "llm": "assistant"}
    history = get_last_messages(conversation_id, n=n)
    turns: list[dict] = []
    for sender, content in history:
        role = role_map.get(sender, "user")
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + content
        else:
            turns.append({"role": role, "content": content})
    return turns


def _get_summary_block(conversation_id: str) -> str:
    summary = get_latest_summary(conversation_id)
    return (
        f"Prior conversation summary:\n{summary}"
        if summary
        else "(none — conversation is short enough to fit in full below)"
    )


def _format_history_flat(conversation_id: str, n: int = 10) -> str:
    """Format history as flat Friend:/You: text for analysis prompts (Step 1/2 of multi-step pipelines)."""
    label = {"friend": "Friend", "user": "You", "llm": "You"}
    history = get_last_messages(conversation_id, n=n)
    return "\n".join(f"{label.get(sender, sender)}: {content}" for sender, content in history)


async def _call_gpt4o_mini(messages: list[dict], temperature: float = 0.7) -> str:
    """
    Call GPT-4o-mini via OpenRouter. Returns the response string with markdown fences stripped.
    Used for all analysis/evaluation steps that need a smart reasoning model.
    """
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": settings.judge_model, "messages": messages, "temperature": temperature}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{settings.base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()
    return clean


# ── Reply model dispatch ──────────────────────────────────────────────────────

REPLY_MODELS = {
    "qwen-local":  {"type": "ollama"},
    "gpt-4o-mini": {"type": "openrouter", "model": "openai/gpt-4o-mini"},
}


async def _call_reply_model(messages: list[dict], reply_model: str) -> str:
    """Dispatch a non-streaming reply call to the right backend based on REPLY_MODELS registry."""
    cfg = REPLY_MODELS.get(reply_model)
    if cfg is None:
        raise ValueError(f"Unknown reply_model: {reply_model!r}. Valid keys: {list(REPLY_MODELS)}")

    if cfg["type"] == "ollama":
        payload = {"model": settings.ollama_model, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
            resp.raise_for_status()
        return resp.json()["message"]["content"]

    model_id = cfg.get("model", settings.model_name)
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.base_url}/chat/completions",
            headers=headers,
            json={"model": model_id, "messages": messages, "temperature": 0.8},
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── single pipeline ───────────────────────────────────────────────────────────

def build_messages_single(conversation_id: str) -> list[dict]:
    """Build the full payload for the single-step pipeline."""
    system = (
        _load_prompt(settings.system_prompt_file)
        .replace("{SUMMARY}", _get_summary_block(conversation_id))
    )
    messages = [{"role": "system", "content": system}, *_build_chat_turns(conversation_id)]
    logger.info(
        "\n" + "=" * 60 + "\nLLM PAYLOAD  conversation_id=%s\n" + "=" * 60 + "\n%s\n" + "=" * 60,
        conversation_id, json.dumps(messages, ensure_ascii=False, indent=2),
    )
    return messages


async def stream_single(conversation_id: str):
    """Async generator: stream Ollama reply for the single-step pipeline."""
    payload = {
        "model": settings.ollama_model,
        "messages": build_messages_single(conversation_id),
        "stream": True,
        "options": {"num_predict": 400},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{settings.ollama_base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done"):
                        yield data["message"]["content"]


async def generate_reply_single(conversation_id: str, reply_model: str = "qwen-local") -> dict:
    """Single-step pipeline: full context → reply in one LLM call."""
    return {"reply": await _call_reply_model(build_messages_single(conversation_id), reply_model)}


# ── two_step pipeline ─────────────────────────────────────────────────────────

async def brainstorm_best_direction(conversation_id: str) -> dict:
    """
    two_step Step 1: GPT-4o-mini analyzes the conversation and picks the single best direction.
    Returns parsed dict with tone, energy, directions, and best.
    """
    messages = [
        {"role": "system", "content": _load_prompt(settings.two_step_brainstorm_file)},
        {"role": "user", "content": f"对话记录：\n{_format_history_flat(conversation_id, n=10)}"},
    ]
    raw = await _call_gpt4o_mini(messages, temperature=0.7)
    brainstorm = json.loads(raw)
    logger.info(
        "two_step Step 1: conversation_id=%s tone=%s best_angle=%s",
        conversation_id, brainstorm.get("tone"), brainstorm.get("best", {}).get("angle"),
    )
    return brainstorm


def build_messages_two_step(conversation_id: str, brainstorm: dict) -> list[dict]:
    """Payload for two_step Step 2: injects the selected direction into the refiner prompt."""
    best = brainstorm.get("best", {})
    direction_block = (
        f"情绪: {brainstorm.get('tone', '未知')}\n"
        f"方向: {best.get('angle', '真诚回应')}\n"
        f"草稿参考: {best.get('draft', '')}"
    )
    system = (
        _load_prompt(settings.two_step_refiner_file)
        .replace("{DIRECTION}", direction_block)
        .replace("{SUMMARY}", _get_summary_block(conversation_id))
    )
    return [{"role": "system", "content": system}, *_build_chat_turns(conversation_id)]


def build_messages_two_step_all(conversation_id: str, brainstorm: dict) -> list[dict]:
    """Payload for two_step_all Step 2: all directions shown, recommended marked."""
    best = brainstorm.get("best", {})
    directions = brainstorm.get("directions", [])
    lines = [f"情绪: {brainstorm.get('tone', '未知')} | 能量: {brainstorm.get('energy', '未知')}", "", "可选方向："]
    for i, d in enumerate(directions, 1):
        marker = " ← 推荐" if d.get("draft") == best.get("draft") else ""
        lines.append(f"{i}. [{d.get('keyword', '')}] {d.get('angle', '')} → {d.get('draft', '')}{marker}")
    lines += ["", "参考推荐方向，用你自己的方式表达。"]
    system = (
        _load_prompt(settings.two_step_refiner_all_file)
        .replace("{DIRECTION}", "\n".join(lines))
        .replace("{SUMMARY}", _get_summary_block(conversation_id))
    )
    return [{"role": "system", "content": system}, *_build_chat_turns(conversation_id)]


async def stream_two_step(conversation_id: str, brainstorm: dict):
    """Async generator: stream Ollama reply for two_step Step 2."""
    payload = {
        "model": settings.ollama_model,
        "messages": build_messages_two_step(conversation_id, brainstorm),
        "stream": True,
        "options": {"num_predict": 400},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{settings.ollama_base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done"):
                        yield data["message"]["content"]


async def stream_two_step_all(conversation_id: str, brainstorm: dict):
    """Async generator: stream Ollama reply for two_step_all Step 2."""
    payload = {
        "model": settings.ollama_model,
        "messages": build_messages_two_step_all(conversation_id, brainstorm),
        "stream": True,
        "options": {"num_predict": 400},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{settings.ollama_base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done"):
                        yield data["message"]["content"]


async def generate_reply_two_step(conversation_id: str, reply_model: str = "qwen-local") -> dict:
    """two_step pipeline: GPT-4o-mini picks best direction → reply_model refines."""
    brainstorm = await brainstorm_best_direction(conversation_id)
    reply = await _call_reply_model(build_messages_two_step(conversation_id, brainstorm), reply_model)
    return {"reply": reply, "brainstorm": brainstorm}


async def generate_reply_two_step_all(conversation_id: str, reply_model: str = "qwen-local") -> dict:
    """two_step_all pipeline: GPT-4o-mini brainstorms all directions → reply_model picks any."""
    brainstorm = await brainstorm_best_direction(conversation_id)
    reply = await _call_reply_model(build_messages_two_step_all(conversation_id, brainstorm), reply_model)
    return {"reply": reply, "brainstorm": brainstorm}


# ── three_step pipeline ───────────────────────────────────────────────────────

async def generate_candidate_directions(conversation_id: str) -> dict:
    """
    three_step Step 1: GPT-4o-mini analyzes the conversation and generates 3 candidate directions.
    Returns {analysis: {emotion, energy, keywords}, directions: [{type, angle, draft}, ...]}.
    Each direction is a different move type: direct, lateral, or probe.
    """
    history_text = _format_history_flat(conversation_id, n=10)
    summary = get_latest_summary(conversation_id) or ""
    prompt = (
        _load_prompt(settings.three_step_analyze_file)
        .replace("{CONVERSATION}", history_text)
        .replace("{SUMMARY}", summary)
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "请分析对话，输出JSON。"},
    ]
    raw = await _call_gpt4o_mini(messages, temperature=0.8)
    result = json.loads(raw)
    logger.info(
        "three_step Step 1: conversation_id=%s emotion=%s directions=%d",
        conversation_id,
        result.get("analysis", {}).get("emotion"),
        len(result.get("directions", [])),
    )
    return result


async def select_best_direction(conversation_id: str, candidates: dict) -> dict:
    """
    three_step Step 2: GPT-4o-mini evaluates the 3 candidates and selects + refines the best one.
    Returns {reasoning, angle, draft}.
    reasoning is logged for observability but NOT forwarded to Step 3 — only angle+draft go forward.
    """
    history_text = _format_history_flat(conversation_id, n=10)
    prompt = (
        _load_prompt(settings.three_step_evaluate_file)
        .replace("{CONVERSATION}", history_text)
        .replace("{STEP1_OUTPUT}", json.dumps(candidates, ensure_ascii=False, indent=2))
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "请评估候选方向，输出JSON。"},
    ]
    raw = await _call_gpt4o_mini(messages, temperature=0.3)
    result = json.loads(raw)
    logger.info(
        "three_step Step 2: conversation_id=%s reasoning=%s angle=%s",
        conversation_id,
        result.get("reasoning", "")[:80],
        result.get("angle"),
    )
    return result


def build_messages_three_step(conversation_id: str, direction: dict) -> list[dict]:
    """
    Payload for three_step Step 3: injects the refined direction brief into the delivery prompt.
    direction = {angle, draft} — reasoning is stripped before this call.
    """
    direction_block = f"方向：{direction.get('angle', '')}\n草稿：{direction.get('draft', '')}"
    system = (
        _load_prompt(settings.three_step_deliver_file)
        .replace("{DIRECTION}", direction_block)
        .replace("{SUMMARY}", _get_summary_block(conversation_id))
    )
    return [{"role": "system", "content": system}, *_build_chat_turns(conversation_id)]


async def stream_three_step(conversation_id: str, direction: dict):
    """Async generator: stream Ollama reply for three_step Step 3."""
    payload = {
        "model": settings.ollama_model,
        "messages": build_messages_three_step(conversation_id, direction),
        "stream": True,
        "options": {"num_predict": 400},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{settings.ollama_base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if not data.get("done"):
                        yield data["message"]["content"]


async def generate_reply_three_step(conversation_id: str, reply_model: str = "qwen-local") -> dict:
    """
    three_step pipeline:
      Step 1 — GPT-4o-mini generates 3 candidate directions (analysis + 3 options)
      Step 2 — GPT-4o-mini evaluates candidates, picks + refines the best (with reasoning)
      Step 3 — reply_model delivers the final reply given the refined direction
    Returns {reply, step1, step2} for full pipeline observability.
    """
    step1 = await generate_candidate_directions(conversation_id)
    step2 = await select_best_direction(conversation_id, step1)
    brief = {"angle": step2["angle"], "draft": step2["draft"]}  # reasoning stays in step2 for logging
    reply = await _call_reply_model(build_messages_three_step(conversation_id, brief), reply_model)
    return {"reply": reply, "step1": step1, "step2": step2}


# ── Summarisation ─────────────────────────────────────────────────────────────

_HISTORY_WINDOW = 20
_SUMMARIZE_EVERY = 5


async def _generate_summary(overflow: list[tuple[str, str]], existing_summary: str) -> str:
    label = {"friend": "Friend", "user": "You", "llm": "You"}
    transcript = "\n".join(f"{label.get(s, s)}: {c}" for s, c in overflow)
    prior_block = f"Existing summary:\n{existing_summary}\n\n" if existing_summary else ""
    prompt = (
        f"{prior_block}New messages to incorporate:\n{transcript}\n\n"
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
    """Trigger a summary update if the conversation has grown beyond the history window."""
    all_messages = get_all_messages(conversation_id)
    total = len(all_messages)
    if total <= _HISTORY_WINDOW:
        return
    overflow_count = total - _HISTORY_WINDOW
    if overflow_count % _SUMMARIZE_EVERY != 0:
        return
    overflow = all_messages[:overflow_count]
    existing_summary = get_latest_summary(conversation_id)
    logger.info(
        "Summarizing conversation_id=%s: %d overflow messages (existing summary: %s)",
        conversation_id, len(overflow), "yes" if existing_summary else "none",
    )
    summary = await _generate_summary(overflow, existing_summary)
    save_summary(conversation_id, summary)
    logger.info("Summary saved for conversation_id=%s", conversation_id)


# ── Judge (LLM-as-evaluator) ──────────────────────────────────────────────────

async def judge_reply(
    message_id: int,
    conversation_id: str,
    candidate_reply: str,
    pipeline: str = "single",
    reply_model: str = "qwen-local",
) -> None:
    """
    Score a candidate reply using GPT-4o-mini. Runs as background task — never raises.
    overall_score is 0-100%: (weighted_avg - 1) / 2 * 100, always valid as rubric changes
    as long as dimension weights sum to 1.0 and scores are on a 1-3 scale.
    """
    label = {"friend": "对方", "user": "你", "llm": "你"}
    history = get_last_messages(conversation_id, n=10)
    history_text = "\n".join(f"{label.get(s, s)}: {c}" for s, c in history)
    user_content = f"对话记录（时间顺序）：\n{history_text}\n\n待评估的建议回复：\n{candidate_reply}"
    messages = [
        {"role": "system", "content": _load_prompt(settings.judge_prompt_file)},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = await _call_gpt4o_mini(messages, temperature=0.0)
        data = json.loads(raw)
        weights = {
            "specificity": 0.20, "need_accuracy": 0.22, "peer_authenticity": 0.20,
            "respect": 0.15, "engagement": 0.13, "originality": 0.10,
        }
        # Normalise to 0-100%: min score=1 maps to 0, max score=3 maps to 100
        weighted_avg = sum(data.get(k, 0) * w for k, w in weights.items())
        overall = round((weighted_avg - 1) / 2 * 100, 1)
        dim_scores = {k: data.get(k) for k in weights}
        save_judge_score(
            message_id=message_id,
            conversation_id=conversation_id,
            overall_score=overall,
            verdict=data.get("verdict"),
            scores_json=json.dumps(dim_scores, ensure_ascii=False),
            justifications=json.dumps(data.get("justifications", {}), ensure_ascii=False),
            rubric_version=settings.judge_prompt_file,
            judge_model=settings.judge_model,
            raw_response=raw,
            pipeline=pipeline,
            reply_model=reply_model,
        )
        logger.info("Judge score saved: message_id=%s overall=%.1f", message_id, overall)
    except Exception:
        logger.warning("judge_reply failed for message_id=%s", message_id, exc_info=True)
