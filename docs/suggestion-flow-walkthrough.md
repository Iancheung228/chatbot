# Suggestion Flow Walkthrough

End-to-end example of the suggestion/accept/reject flow.
Friend message used: **"你今晚有空吗？"**

---

## Step 1 — User pastes friend's message, clicks "Send as Friend"

**Frontend (`frontend/app.py`):**
- Calls `save_message()` directly → inserts friend row
- Sets `pending_llm_call = True`, clears suggestions and input
- Calls `st.rerun()`

**DB after this step:**
```
id=1  sender='friend'  content='你今晚有空吗？'  sent=1  source=null
```

Chat UI shows the friend's bubble. Text area clears.

---

## Step 2 — Pending LLM call fires

**Frontend (`frontend/app.py`):**
- `POST /suggest_reply` with `conversation_id`

**Backend (`api/main.py` → `api/llm.py`):**
- `build_api_payload()` fetches last 20 `sent=1` rows → only the friend message
- Sends to LLM, streams back — **writes nothing to DB**

**Frontend (after stream completes):**
- Captures full text via `st.write_stream`
- `POST /log_suggestion` → gets back `suggestion_id`
- Appends `{"text": "有空！你想干嘛？", "suggestion_id": 2}` to `session_state["suggestions"]`
- Calls `st.rerun()`

**Backend (`api/main.py` → `api/db.py`):**
- `log_llm_suggestion()` inserts row with `sender='llm', sent=0`

**DB after this step:**
```
id=1  sender='friend'  content='你今晚有空吗？'    sent=1
id=2  sender='llm'     content='有空！你想干嘛？'  sent=0  ← pending
```

Suggestion panel appears between chat and input with one card.

---

## Step 3 — User clicks Regenerate

Same as Step 2. LLM is called again with the same DB context (friend message
is still the last `sent=1` row). New suggestion appended to the panel.

**DB after this step:**
```
id=1  sender='friend'  content='你今晚有空吗？'           sent=1
id=2  sender='llm'     content='有空！你想干嘛？'         sent=0
id=3  sender='llm'     content='今晚有点事，明天呢？'     sent=0
```

Suggestion panel shows two cards. Counter says "2 suggestions generated."

---

## Step 4a — User accepts suggestion exactly (no edits)

User clicks "Use this ↑" on suggestion #2 → text area pre-fills with
`今晚有点事，明天呢？` → clicks **Send as User** without changing anything.

**Frontend — source detection:**
```
final text == prefilled text  →  source = "llm_accepted"
```

**`POST /send_user_message`:**
```json
{
  "conversation_id": "...",
  "content": "今晚有点事，明天呢？",
  "source": "llm_accepted",
  "suggestion_id": 3
}
```

**Backend:**
- `mark_suggestion_sent(3)` → `UPDATE messages SET sent=1 WHERE id=3 AND sender='llm' AND sent=0`

**DB after this step:**
```
id=1  sender='friend'  content='你今晚有空吗？'           sent=1
id=2  sender='llm'     content='有空！你想干嘛？'         sent=0  ← rejected
id=3  sender='llm'     content='今晚有点事，明天呢？'     sent=1  ← accepted
```

Chat UI shows `今晚有点事，明天呢？` as a right bubble
(`sender='llm'` with `sent=1` renders the same as a user bubble).

---

## Step 4b — User edits the suggestion before sending

User clicks "Use this ↑" on #2 → text area fills → user changes it to
`今晚有点事，后天呢？` → clicks **Send as User**.

**Frontend — source detection:**
```
final text != prefilled text  →  source = "llm_modified"
```

**`POST /send_user_message`:**
```json
{
  "conversation_id": "...",
  "content": "今晚有点事，后天呢？",
  "source": "llm_modified"
}
```

**Backend:**
- `save_message(conv_id, "user", "今晚有点事，后天呢？", sent=1, source="llm_modified")`

**DB after this step:**
```
id=2  sender='llm'   content='有空！你想干嘛？'         sent=0  ← rejected
id=3  sender='llm'   content='今晚有点事，明天呢？'     sent=0  ← original preserved
id=4  sender='user'  content='今晚有点事，后天呢？'     sent=1  source='llm_modified'
```

The original LLM text is preserved alongside the user's edit — useful for
training data analysis (diff what the model suggested vs. what user sent).

---

## Step 4c — User ignores all suggestions, types manually

No "Use this ↑" clicked → `prefilled_suggestion` is `None`.

**Frontend — source detection:**
```
prefilled_suggestion is None  →  source = "manual"
```

**`POST /send_user_message`:**
```json
{
  "conversation_id": "...",
  "content": "没空",
  "source": "manual"
}
```

**DB after this step:**
```
id=2  sender='llm'   sent=0  ← rejected
id=3  sender='llm'   sent=0  ← rejected
id=4  sender='user'  content='没空'  sent=1  source='manual'
```

---

## What the next LLM call sees

When the friend sends another message, `build_api_payload()` runs
`get_last_messages()` which filters `WHERE sent=1`:

```
("friend", "你今晚有空吗？")      →  ChatML role: "user"
("llm",    "今晚有点事，明天呢？") →  ChatML role: "assistant"  (Step 4a path)
```

Both `sender='user'` and `sender='llm'` with `sent=1` map to `"assistant"`.
The two rejected `sent=0` suggestions are completely invisible to the LLM.

---

## DB schema reference

| column | values | meaning |
|--------|--------|---------|
| `sender` | `'friend'` / `'user'` / `'llm'` | who produced this message |
| `sent` | `1` / `0` | 1 = confirmed/shown in chat, 0 = pending/rejected suggestion |
| `source` | `null` / `'manual'` / `'llm_accepted'` / `'llm_modified'` | how the user's reply was produced |

**Reading the data:**
- `sender='llm', sent=0` → suggestion that was generated but not used
- `sender='llm', sent=1` → suggestion accepted exactly as-is
- `sender='user', sent=1, source='llm_modified'` → user started from a suggestion, edited it
- `sender='user', sent=1, source='manual'` → user typed from scratch
