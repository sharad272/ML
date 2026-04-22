from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from chatbot_backend import chatbot

# If checkpoint is there then we need to pass thread_id to the frontend too!
CONFIG = {"configurable":{"thread_id":1}}

user_input = st.chat_input('Type here')
# st.session_state -> dict -> when pressing enter (except manually refreshing the page)


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# displaying conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


if user_input:

    #Adding msg to message history of the user
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)


    initial_state = {
        'messages' : [HumanMessage(content = user_input)]
    }
    
    # Calling LLM 
   # response = chatbot.invoke(initial_state,config= CONFIG)['messages'][-1].content

    # To make the response in stream mode
    


    #Adding msg to message history of the AI    

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {
                    'messages' : [HumanMessage(content = user_input ) ]
                },
                config = CONFIG,
                stream_mode = 'messages'
            )
        )

    st.session_state['message_history'].append({'role':'Assistant', 'content': ai_message})
    