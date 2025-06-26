import streamlit as st
from vector_debug import ask_debug_agent

st.set_page_config(page_title="Microbit Debug Assistant", layout="centered")
st.title("🤖 Microbit Debug Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask me a Microbit issue")

if user_input:
    with st.spinner("Thinking..."):
        reply = ask_debug_agent(user_input)
        st.session_state.chat.append(("user", user_input))
        st.session_state.chat.append(("agent", reply))

for role, msg in st.session_state.chat:
    st.chat_message("🧑‍💻" if role == "user" else "🤖").write(msg)