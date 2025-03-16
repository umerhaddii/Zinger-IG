import streamlit as st
from app import create_chat

st.title("Zinger Assistant Chatbot")
st.write("Welcome to the Zinger Assistant Chatbot! Type your message below to start a conversation.")

# Initialize chat in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = create_chat()
if "messages" not in st.session_state:
    st.session_state.messages = []


# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response
    response = st.session_state.conversation.predict(input=prompt)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

