import streamlit as st
from vector_debug import ask_debug_agent, display_initial_guidelines

st.set_page_config(page_title="Microbit Debug Assistant", layout="centered")
st.title("Debugging Assistant")

# Initialize chat with guidelines
if "chat" not in st.session_state:
    st.session_state.chat = []
    # Add initial guidelines message
    guidelines = display_initial_guidelines()
    st.session_state.chat.append(("assistant", guidelines))

user_input = st.chat_input("Ask me a Microbit issue")

if user_input:
    with st.spinner("Thinking..."):
        reply = ask_debug_agent(user_input)
        st.session_state.chat.append(("user", user_input))
        st.session_state.chat.append(("assistant", reply))

for role, msg in st.session_state.chat:
    if role == "user":
        st.chat_message("🧑‍💻").write(msg)
    else:
        st.chat_message("🤖").markdown(msg)