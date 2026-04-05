import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

import json
import streamlit as st
import requests
import uuid

st.title("Message Coach 🤖💬")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = str(uuid.uuid4())

if "suggestions" not in st.session_state:
    # Each entry: {"text": str, "suggestion_id": int}
    st.session_state["suggestions"] = []

if "prefilled_suggestion" not in st.session_state:
    # Set when user clicks "Use this ↑"; cleared after Send as User
    # {"text": str, "suggestion_id": int} | None
    st.session_state["prefilled_suggestion"] = None

if "pending_llm_call" not in st.session_state:
    # True when we need to call the LLM (after Send as Friend or Regenerate)
    st.session_state["pending_llm_call"] = False

if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False

if "num_input_box" not in st.session_state:
    st.session_state["num_input_box"] = ""

# ---------------------------------------------------------------------------
# Clear text area on the run BEFORE the widget is instantiated
# ---------------------------------------------------------------------------
if st.session_state["clear_input"]:
    st.session_state["num_input_box"] = ""
    st.session_state["clear_input"] = False

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
def fetch_history(conversation_id):
    try:
        response = requests.get(
            f"{API_BASE}/get_history",
            params={"conversation_id": conversation_id},
        )
        if response.status_code == 200:
            return response.json().get("messages", [])
    except Exception:
        pass
    return []


def display_chat_ui(conversation_id, placeholder):
    with placeholder.container():
        messages = fetch_history(conversation_id)
        for msg in messages:
            sender = msg["sender"]
            content = msg["content"]
            # sender='user' (manual/modified) and sender='llm' (accepted) both
            # appear on the right — both represent what the user chose to send.
            if sender in ("user", "llm"):
                st.markdown(
                    f"<div style='text-align:right; background-color:#DCF8C6; "
                    f"padding:8px 16px; border-radius:18px; margin:5px 0 5px 40px; "
                    f"max-width:70%; float:right;'>{content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:left; background-color:#F1F0F0; "
                    f"padding:8px 16px; border-radius:18px; margin:5px 40px 5px 0; "
                    f"max-width:70%; float:left;'>{content}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Layout: chat → suggestions → input → buttons
# ---------------------------------------------------------------------------
chat_placeholder = st.empty()
display_chat_ui(st.session_state["conversation_id"], chat_placeholder)

# --- Suggestion panel ---
if st.session_state["suggestions"]:
    st.markdown("---")
    st.markdown("**💡 Suggestions**")
    for idx, suggestion in enumerate(st.session_state["suggestions"]):
        with st.container(border=True):
            col_text, col_btn = st.columns([4, 1])
            with col_text:
                st.write(f"**#{idx + 1}** {suggestion['text']}")
            with col_btn:
                def _use_suggestion(text=suggestion["text"], sid=suggestion["suggestion_id"]):
                    st.session_state["num_input_box"] = text
                    st.session_state["prefilled_suggestion"] = {"text": text, "suggestion_id": sid}
                st.button(
                    "Use this ↑",
                    key=f"use_suggestion_{idx}",
                    on_click=_use_suggestion,
                )

    regen_col, counter_col = st.columns([1, 3])
    with regen_col:
        if st.button("🔄 Regenerate"):
            st.session_state["pending_llm_call"] = True
    with counter_col:
        n = len(st.session_state["suggestions"])
        st.caption(f"{n} suggestion{'s' if n != 1 else ''} generated")
    st.markdown("---")

# --- Input box + send buttons (form so button click commits text area value) ---
with st.form("message_form", clear_on_submit=False):
    st.text_area(
        "Paste the message you received or edit a suggestion:",
        key="num_input_box",
    )
    col1, col2 = st.columns(2)
    send_as_friend = col1.form_submit_button("Send as Friend 👤")
    send_as_user   = col2.form_submit_button("Send as User 💬")

# ---------------------------------------------------------------------------
# Handle: Send as Friend
# ---------------------------------------------------------------------------
if send_as_friend:
    message_text = st.session_state["num_input_box"].strip()
    try:
        resp = requests.post(
            f"{API_BASE}/friend_message",
            json={"conversation_id": st.session_state["conversation_id"], "content": message_text},
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            st.error(f"Failed to save friend message: {resp.text}")
            st.stop()
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        st.stop()
    # Clear any suggestions from the previous round
    st.session_state["suggestions"] = []
    st.session_state["prefilled_suggestion"] = None
    st.session_state["pending_llm_call"] = True
    st.session_state["clear_input"] = True
    st.rerun()

# ---------------------------------------------------------------------------
# Handle: Send as User
# ---------------------------------------------------------------------------
if send_as_user:
    message_text = st.session_state["num_input_box"].strip()
    if message_text:
        conv_id = st.session_state["conversation_id"]
        pre = st.session_state.get("prefilled_suggestion")

        # Determine source
        if pre is None:
            source = "manual"
            payload = {"conversation_id": conv_id, "content": message_text, "source": source}
        elif message_text == pre["text"]:
            source = "llm_accepted"
            payload = {
                "conversation_id": conv_id,
                "content": message_text,
                "source": source,
                "suggestion_id": pre["suggestion_id"],
            }
        else:
            source = "llm_modified"
            payload = {"conversation_id": conv_id, "content": message_text, "source": source}

        try:
            resp = requests.post(f"{API_BASE}/send_user_message", json=payload, timeout=10)
            if resp.status_code == 404 and source == "llm_accepted":
                # Suggestion row not found (e.g. logging failed earlier) — fall back to manual
                fallback = {"conversation_id": conv_id, "content": message_text, "source": "manual"}
                resp = requests.post(f"{API_BASE}/send_user_message", json=fallback, timeout=10)
            if resp.status_code not in (200, 201):
                st.error(f"Failed to save message: {resp.text}")
                st.stop()
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
            st.stop()

        # Clear suggestions and input for next round
        st.session_state["suggestions"] = []
        st.session_state["prefilled_suggestion"] = None
        st.session_state["pending_llm_call"] = False
        st.session_state["clear_input"] = True
        st.rerun()
    else:
        st.warning("Please enter a message before sending.")

# ---------------------------------------------------------------------------
# Handle: pending LLM call (Send as Friend or Regenerate triggered this)
# ---------------------------------------------------------------------------
if st.session_state["pending_llm_call"]:
    st.session_state["pending_llm_call"] = False
    conv_id = st.session_state["conversation_id"]
    suggestion_text = ""
    suggestion_id = None

    with st.spinner("Generating suggestion..."):
        try:
            response = requests.post(
                f"{API_BASE}/suggest_reply",
                json={"conversation_id": conv_id},
                stream=True,
                timeout=120,
            )
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    # OpenRouter / vLLM — single JSON response with suggestion_id included
                    data = response.json()
                    suggestion_text = data.get("reply", "")
                    suggestion_id = data.get("suggestion_id")
                    if suggestion_text:
                        st.write(suggestion_text)
                else:
                    # Ollama NDJSON streaming path
                    # Each line: {"chunk": "..."} or final {"done": true, "suggestion_id": N}
                    _state = {"suggestion_id": None}

                    def _text_chunks():
                        for line in response.iter_lines():
                            if not line:
                                continue
                            data = json.loads(line)
                            if data.get("done"):
                                _state["suggestion_id"] = data.get("suggestion_id")
                            else:
                                yield data.get("chunk", "")

                    suggestion_text = st.write_stream(_text_chunks())
                    suggestion_id = _state["suggestion_id"]
            else:
                st.error(f"Backend error: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

    if suggestion_text and suggestion_text.strip():
        st.session_state["suggestions"].append(
            {"text": suggestion_text.strip(), "suggestion_id": suggestion_id}
        )

    st.rerun()

# ---------------------------------------------------------------------------
# Other controls
# ---------------------------------------------------------------------------
def start_new_conversation():
    st.session_state["conversation_id"] = str(uuid.uuid4())
    st.session_state["num_input_box"] = ""
    st.session_state["suggestions"] = []
    st.session_state["prefilled_suggestion"] = None
    st.session_state["pending_llm_call"] = False

st.button("Start New Conversation", on_click=start_new_conversation)

