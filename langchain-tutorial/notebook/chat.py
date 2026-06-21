from fastapi import FastAPI
from langchain_core.tools import tool
from langchain_core.tools import StructuredTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
load_dotenv()


#calcultor tools
@tool
def add(a:int, b:int)->int:
    """Add Two numbers"""
    return a+b

@tool
def subtract(a:int, b:int)->int:
    """Subtract Two numbers"""
    return a-b

@tool
def multiply(a:int, b:int)->int:
    """Multiply Two numbers"""
    return a*b

@tool
def divide(a:int, b:int)->int:
    """Divide Two numbers"""
    return a/b


#searcg tool
api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=500)
sr_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
@tool
def search(q:str)->str:
    """Search Wikipedia for factual information,
        populations, dates, people, places,
        companies, statistics, and general knowledge.
        Use this whenever information must be looked up.
    """
    return sr_tool.invoke({"query":q})


#llm model
llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)


#agent
agent = llm.bind_tools([add, subtract, multiply, divide, search],tool_choice="auto")

#mapping tools
tools = {
    search.name : search,
    add.name : add,
    subtract.name : subtract,
    multiply.name : multiply,
    divide.name : divide    
}

SYSTEM_PROMPT = """
You are an AI assistant.

When a user asks for factual information,
ALWAYS use the search tool.

When a user asks for calculations,
use calculator tools.

You may use multiple tools in sequence.
"""


#chat function
def chat(query:str):
    SystemMessage(content=SYSTEM_PROMPT),
    messages = [HumanMessage(content=query)]
    
    while True:
        response = agent.invoke(messages)
        print(response)
        print(response.tool_calls)
        
        if not response.tool_calls:
            return response.content
    
        messages.append(response)
        
        for tool_call in response.tool_calls:
            result = tools[
                tool_call["name"]
            ].invoke(
                tool_call['args']
            )
            
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id = tool_call['id']
                )
            )
            
#chat schema
class ChatSchema(BaseModel):
    message: str
    
#app initialisation
app = FastAPI()
    
#chat endpoint
@app.post("/chat")
async def chat_endpoin(req:ChatSchema):
    response = chat(req.message)
    return{"response":response}