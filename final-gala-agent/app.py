import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

SYSTEM_POLICY = """You are Alfred, a tool-using assistant.

Routing policy (use the FEWEST calls necessary):
- If the user asks about a gala guest by NAME or RELATION → call tool: guest_info_tool with the user's exact question.
  - Normalize titles like Lady/Sir/Dr/Prof when matching, but pass the raw query to the tool.
- If the user asks “who is <company/person>” or general facts not limited to the guest list → call tool: search_tool.
- If the user asks for “most popular/top/downloads model” of an org on the model hub → call tool: hub_stats_tool
  (optionally call search_tool first to disambiguate the org name).
- Only use weather_info_tool for weather/forecast queries.

Answering policy:
- Summarize STRICTLY from tool outputs; do NOT add outside knowledge.
- Write in your own words (don’t copy tool text verbatim).
- Natural, friendly, concise: 1–2 sentences, no bullets or headings, in the user’s language.
- For guest queries, focus on the Description and include the name if helpful.
- If guest_info_tool returns no relevant match, reply exactly: No matching guest information found.
- If other tools return nothing useful, reply exactly: I couldn’t find this with the available tools.

Operational rules:
- Always include the user’s exact question as the tool input.
- Prefer a single call; chain tools only when needed to resolve ambiguity.
"""

# Initialize the web search tool
search_tool = DuckDuckGoSearchRun()

# Generate the chat interface, including the tools
# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
# )

chat = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
)

# chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    msgs = [SystemMessage(content=SYSTEM_POLICY)] + state["messages"]
    return { "messages": [chat_with_tools.invoke(msgs)]}

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

# First interaction
response = alfred.invoke({"messages": [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")]})


print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
print()

# Second interaction (referencing the first)
response = alfred.invoke({"messages": response["messages"] + [HumanMessage(content="What projects is she currently working on?")]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)