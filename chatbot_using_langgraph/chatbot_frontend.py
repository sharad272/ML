from langchain_core.messages import HumanMessage
import streamlit as st
from chatbot_backend import chatbot

CONFIG = {"configurable": {"thread_id": 1}}

user_input = st.chat_input('Type here')

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if user_input:

    # Store user message
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.markdown(user_input)

    # Call LLM
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]},
        config=CONFIG
    )['messages'][-1].content

    # Display assistant response
    with st.chat_message('assistant'):
        st.markdown(response)

    # Store assistant response (FIXED)
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': response
    })