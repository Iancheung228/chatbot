import streamlit as st
import requests
import uuid

st.title("Message Coach 🤖💬")

# Text input for user to paste a message
user_message = st.text_area("Paste the message you received:")


# Initialize conversation ID in session state
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = str(uuid.uuid4())

# To reset conversation
if st.button("Start New Conversation"):
    st.session_state["conversation_id"] = str(uuid.uuid4())



if st.button("Get Reply Suggestion"):
    if user_message.strip():
        try:
            # Send the message to your FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/suggest_reply",
                json={
                    "message": user_message,
                    "conversation_id": st.session_state["conversation_id"]
                    }
            )
            if response.status_code == 200:
                data = response.json()
                st.write(data["motivation_analysis"])
                st.success("Here’s a suggestion:")
                st.write(data["suggestions"])
            else:
                st.error(f"Backend error: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
    else:
        st.warning("Please paste a message first.")
