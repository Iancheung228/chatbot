import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

import streamlit as st
import requests
import uuid

st.title("Message Coach 🤖💬")

if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = str(uuid.uuid4())

# Use a flag to clear the input box
if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False

# if st.session_state["clear_input"]:
#     st.session_state["input_box"] = ""
#     st.session_state["clear_input"] = False

def fetch_history(conversation_id):
    try:
        response = requests.get(
            f"{API_BASE}/get_history",
            params={"conversation_id": conversation_id}
        )
        if response.status_code == 200:
            return response.json().get("messages", [])
    except Exception:
        pass
    return []

## write to chat UI:
def display_chat_ui(conversation_id, placeholder):
    with placeholder.container():  # fill reserved slot at the top
        messages = fetch_history(conversation_id)
        for msg in messages:
            sender = msg["sender"]
            content = msg["content"]
            if sender == "user":
                st.markdown(
                    f"<div style='text-align:right; background-color:#DCF8C6; "
                    f"padding:8px 16px; border-radius:18px; margin:5px 0 5px 40px; "
                    f"max-width:70%; float:right;'>{content}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align:left; background-color:#F1F0F0; "
                    f"padding:8px 16px; border-radius:18px; margin:5px 40px 5px 0; "
                    f"max-width:70%; float:left;'>{content}</div>",
                    unsafe_allow_html=True
                )
        st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)


if "input_box" not in st.session_state:
    st.session_state["input_box"] = ""

# 🔹 Placeholder for chat UI at the top
chat_placeholder = st.empty()

# handle_change is not needed now
def handle_change():
    # This runs whenever the text_area content changes
    st.session_state.input_box = st.session_state.num_input_box

#When you type in the box, Streamlit automatically updates st.session_state["input_box"] with your input.
st.text_area( #creates a text input box for you to type
    "Paste the message you received:",
    key="num_input_box", # links the widget's value to st.session_state["input_box"]
    #on_change=handle_change,  # callback function
    #value=st.session_state["input_box"]
)


# --- Add two buttons instead of one ---
col1, col2 = st.columns(2)
is_empty = not st.session_state.get("num_input_box", "").strip()
send_as_friend = col1.button("Send as Friend 👤", disabled=is_empty)
send_as_user = col2.button("Send as User 💬", disabled=is_empty)

# Determine sender based on which button is clicked
sender = None
if send_as_friend:
    sender = "friend"
elif send_as_user:
    sender = "user"

# --- Only run if a button was clicked and message not empty ---
if sender and st.session_state.num_input_box.strip():
    from api.db import init_db, save_message


    # 1️⃣ Save the message with correct sender
    save_message(st.session_state["conversation_id"], sender, st.session_state.num_input_box)

    # 2️⃣ Refresh the chat UI immediately
    display_chat_ui(st.session_state["conversation_id"], chat_placeholder)

    # 3️⃣ If it's a friend message, call the LLM suggestion API
    if sender == "friend":
        try:
            response = requests.post(
                f"{API_BASE}/suggest_reply",
                json={
                    "message": st.session_state.num_input_box,
                    "conversation_id": st.session_state["conversation_id"],
                },
                stream=True,
                timeout=120,
            )
            if response.status_code == 200:
                st.success("Suggested reply:")
                st.write_stream(
                    chunk for chunk in response.iter_content(
                        chunk_size=None, decode_unicode=True
                    ) if chunk
                )
                st.session_state["clear_input"] = True
                st.session_state["input_box"] = ""
            else:
                st.error(f"Backend error: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
    else:
        # 👤 User message logic: maybe you don't want to call LLM here
        st.info("✅ User message added (no LLM call made).")

elif sender:  # If button clicked but no text
    st.warning("Please enter a message before sending.")
# if st.button("Get Reply Suggestion"):
#     if st.session_state.num_input_box.strip(): # only proceed if user_message is not empty
#         # display friend's msg in chat UI immediately
#         from api.db import init_db, save_message
#         # Save the user message to the database
#         save_message(st.session_state["conversation_id"], "friend", st.session_state.num_input_box)
#         display_chat_ui(st.session_state["conversation_id"], chat_placeholder)
        
#         # calls llm
#         try:
#             response = requests.post(
#                 f"{API_BASE}/suggest_reply",
#                 json={
#                     "message": st.session_state.num_input_box,
#                     "conversation_id": st.session_state["conversation_id"]
#                 }
#             )
#             if response.status_code == 200:
#                 data = response.json()
#                 st.write(data["motivation_analysis"])
#                 st.success("Here’s a suggestion:")
#                 st.write(data["suggestions"])
#                 st.session_state["clear_input"] = True # Set flag to clear input box on next rerun
#                 st.session_state["input_box"] = ""
#             else:
#                 st.error(f"Backend error: {response.text}")
#         except Exception as e:
#             st.error(f"Error connecting to backend: {e}")
#     else:
#         st.warning("Please paste a message first.")
    
# 🔹 Always render chat history at the top
display_chat_ui(st.session_state["conversation_id"], chat_placeholder)


def start_new_conversation():
    st.session_state["conversation_id"] = str(uuid.uuid4())
    st.session_state["num_input_box"] = ""
    st.session_state["messages"] = []

st.button("Start New Conversation", on_click=start_new_conversation)    
# if st.button("Start New Conversation"):
#     st.session_state["conversation_id"] = str(uuid.uuid4())
#     st.session_state["input_box"] = ""  
#     st.session_state["clear_input"] = False


## I am using num_input_box so that when i hit start new convo, i dont get error.
# st.session_state.input_box cannot be modified after the widget with key input_box is instantiated.