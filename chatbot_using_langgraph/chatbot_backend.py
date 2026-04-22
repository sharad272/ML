from langchain_core import messages
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, List,Annotated
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver # To save this in RAM
from langchain_core.messages  import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages


load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
if not token:
    raise ValueError("Set HUGGINGFACE_API_KEY or HUGGINGFACEHUB_API_TOKEN in .env")
os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=token
)

model = ChatHuggingFace(llm = llm)


class ChatState(TypedDict):
      messages: Annotated[List[BaseMessage], add_messages]
    

def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {
        "messages": [AIMessage(content= response.content)]
    }



checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

# If checkpoint is there then we ned to pass thread_id to the frontend too!
chatbot = graph.compile(checkpointer=checkpointer)








