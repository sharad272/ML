from langchain_core.messages import HumanMessage
import streamlit as st
from chatbot_backend import chatbot

#CONFIG = {"configurable": {"thread_id": 1}}

user_input = st.chat_input('Type here')

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Handle new input FIRST
if user_input:
    # Add user message
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    # Call LLM
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]}
    )['messages'][-1].content

    # Add assistant message
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': response
    })


# Display history (ONLY place where rendering happens)
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])