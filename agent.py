# ==============================================================================
# 0. IMPORT NECESSARY LIBRARIES
# ==============================================================================
# We start by importing all the tools we'll need for this project.

import os
from typing import Literal, List
from langchain_groq import ChatGroq
from fastapi import FastAPI, Form
from fastapi.responses import ORJSONResponse
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
import uvicorn

# ==============================================================================
# 1. INITIAL SETUP AND CONFIGURATION
# ==============================================================================
"""
In this section, we load our environment variables, initialize the FastAPI
application, and set up the connection to the Groq API with our chosen LLM.
This is the foundational setup for our entire application.
"""

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Multi-Agent API",
    description="A multi-agent LangGraph example with improved routing logic.",
    version="3.0",
    default_response_class=ORJSONResponse,
)

# --- Environment and API Key Setup ---
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in .env file")

os.environ["TAVILY_API_KEY"] = tavily_api_key

# --- LLM and Tools Initialization ---
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0)
tools = [TavilySearchResults(max_results=2)]


# ==============================================================================
# 2. PYDANTIC MODELS FOR DATA STRUCTURE
# ==============================================================================
"""
Pydantic models define the "shape" of our data. They ensure that data,
especially the output from the LLM, is in a predictable, structured format.
This is crucial for building reliable applications.
"""


# --- Pydantic Models for Structured Data ---
class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]
    reason: str = Field(description="A brief explanation for the routing decision.")

class StepResponse(BaseModel):
    step: str
    role: str
    content: str
    timestamp: str

class ChatResponse(BaseModel):
    status: str
    original_query: str
    workflow_steps: List[StepResponse]
    final_answer: str
    completed_at: str


# ==============================================================================
# 3. LANGGRAPH STATE DEFINITION
# ==============================================================================
"""
The "State" is the memory of our graph. It's an object that gets passed
from node to node. We use `MessagesState` to automatically manage a list of
messages (the conversation history) and add our own `next` field to control
the flow of the graph.
"""

# --- Graph State Definition ---
class AgentState(MessagesState):
    next: str


# ==============================================================================
# 4. AGENT NODE DEFINITIONS
# ==============================================================================
"""
Each function below represents a "node" in our graph. A node is a unit of
work, essentially an agent with a specific role and set of instructions.
Each node receives the current state, performs its task, and returns an
update to the state.
"""

# --- Agent Node Definitions ---
def supervisor_node(state: AgentState) -> dict:
    """
    Decides the next action. This node is a strict router and does not answer questions itself.
    """
    # FIX: Updated prompt to force routing for all factual questions.
    system_prompt = (
        "You are a supervisor. Your role is to analyze the user's latest query and route it to the appropriate specialist agent. "
        "Your decision is final. Do not attempt to answer the question yourself, only classify it.\n"
        "Route to the 'researcher' for any questions that involve facts, data, or general knowledge.\n"
        "Route to the 'coder' for questions related to programming, algorithms, or mathematical calculations.\n"
        "Route to 'FINISH' ONLY if the user is saying hello, thank you, or making a simple conversational remark."
    )
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    structured_llm = llm.with_structured_output(SupervisorDecision)
    response = structured_llm.invoke(messages)
    
    return {
        "messages": [HumanMessage(content=response.reason, name="supervisor")],
        "next": response.next
    }

def researcher_node(state: AgentState) -> dict:
    """Handles research-oriented tasks by directly answering the user's last message."""
    # The researcher's job is to answer the original user query.
    user_query = state['messages'][0].content
    system_prompt = f"You are a senior researcher. Provide a direct and concise answer to the following user query: '{user_query}'"
    
    # We invoke the LLM with just the system prompt to get a direct answer.
    result = llm.invoke(system_prompt)
    result.name = "researcher" 
    return {"messages": [result]}

def coder_node(state: AgentState) -> dict:
    """Handles coding and calculation tasks."""
    system_prompt = "You are an expert programmer. Provide a clear solution, including code snippets in Python where applicable."
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    result = llm.invoke(messages)
    result.name = "coder"
    return {"messages": [result]}

# NEW: A simple node to handle conversational finishes.
def answerer_node(state: AgentState) -> dict:
    """Provides a final, conversational answer."""
    system_prompt = "You are a helpful assistant. Respond to the user's last message in a friendly, conversational way."
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    result = llm.invoke(messages)
    result.name = "assistant"
    return {"messages": [result]}



# ==============================================================================
# 5. GRAPH CONSTRUCTION
# ==============================================================================
"""
Here, we assemble the nodes into a coherent workflow using LangGraph.
We define the nodes and the "edges" (the paths) that connect them. This
creates the state machine that powers our multi-agent system.
"""



# --- Graph Construction ---
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)
builder.add_node("answerer", answerer_node) # Add the new node

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "researcher": "researcher",
        "coder": "coder",
        "FINISH": "answerer"  # FIX: Route FINISH to the new answerer node
    }
)

builder.add_edge("researcher", END)
builder.add_edge("coder", END)
builder.add_edge("answerer", END) # The answerer node also ends the graph

graph = builder.compile()



# ==============================================================================
# 6. FASTAPI ENDPOINT DEFINITION
# ==============================================================================
"""
This section defines our API endpoints. The `/chat/` endpoint is the main
interaction point where a user can send a message and get a response from
our LangGraph agent.
"""

# --- FastAPI Endpoint Definition ---
@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(message: str = Form(...)):
    inputs = {"messages": [HumanMessage(content=message, name="user")]}
    workflow_steps = []

    for output in graph.stream(inputs):
        for node_name, state_update in output.items():
            if "messages" in state_update:
                latest_message = state_update["messages"][-1]
                step = StepResponse(
                    step=node_name,
                    role=latest_message.name or "assistant",
                    content=str(latest_message.content),
                    timestamp=datetime.now().isoformat()
                )
                workflow_steps.append(step)
    
    # The final answer is now from the last relevant agent that is not the supervisor.
    final_answer = next(
        (step.content for step in reversed(workflow_steps) if step.role in ["researcher", "coder", "assistant"]),
        "No answer was generated."
    )

    return ChatResponse(
        status="completed",
        original_query=message,
        workflow_steps=workflow_steps,
        final_answer=final_answer,
        completed_at=datetime.now().isoformat()
    )

@app.get("/")
async def home():
    return {"message": "Welcome! Use the /chat/ endpoint with a 'message' parameter to start."}


# ==============================================================================
# 7. MAIN EXECUTION BLOCK
# ==============================================================================
"""
This block allows us to run the FastAPI application directly using Uvicorn,
a high-performance ASGI server. It's the standard way to run FastAPI apps
for development.
"""

# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)