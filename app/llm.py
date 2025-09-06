import httpx
import json
from .config import OPENROUTER_API_KEY, BASE_URL

SYSTEM_PROMPT = """
You are an emotionally intelligent coach/ relationship guru. The user is coming to you for help crafting appropriate replies to messages they receive (from a girl they like).

Return your analysis in a field called "motivation_analysis".

Then, help users craft thoughtful replies to messages they receive. Use the tone of a 25 y.o male. Try to sound like a human, not a robot. Avoid generic responses.

Use the tone of a 25 y.o male. Try to sound like a human, not a robot. Avoid generic responses.

The user pastes a message they received.
Suggest 3 possible replies in different tones:
- Warm
- Playful
- Formal
For each reply, include a short explanation of why it works.
Return responses in JSON-like structure.

Return your response in this JSON structure:
{
  "motivation_analysis": "The sender is likely looking for a compliment and to share their excitement.",
  "suggestions": [
    {
      "tone": "Warm",
      "reply": "...",
      "explanation": "..."
    },
    ...
  ]
}

Example input message:
"Today I won the lottery at our company raffle, I got a headphone!"
Example output (partial):
{
  "suggestions": [
    {
      "tone": "Warm",
      "reply": "Wowww, im so jealous, I have never won anything at the raffle... This is your year of fate so you must be really lucky! Wow I am so flattered your sharing your good luck with me ;)",
      "explanation": "This reply is warm and enthusiastic, showing genuine happiness for the sender's good fortune."
    }...
}
"""


def build_prompt(conversation_id, user_message):
    from app.db import get_last_messages, get_latest_summary

    summary = get_latest_summary(conversation_id)
    last_messages = get_last_messages(conversation_id, n=10)
    context = "\n".join([f"{sender}: {content}" for sender, content in last_messages])

    prompt = f"""
    Conversation summary so far :
    {summary}

    Recent messages:
    {context}

    New message:
    {user_message}

    {SYSTEM_PROMPT}
    """
    return prompt



async def generate_replies(user_message: str, conversation_id: str, model: str = "openai/gpt-4o-mini"):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    #print( build_prompt(conversation_id, user_message))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_prompt(conversation_id, user_message)},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.8
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Parse the content as JSON
    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        # If parsing fails, return the raw content
        return content


