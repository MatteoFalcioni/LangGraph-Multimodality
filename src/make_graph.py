from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

from state import MultiState
from utils import prepare_multimodal_message
from prompts.multimodal import multimodal_prompt
from prompts.coding import coding_prompt
from tools.coding import code_exec_tool

load_dotenv()

multimodal_model = ChatOpenAI(model="gpt-4o")
multimodal_agent = create_react_agent(
    model=multimodal_model,
    tools=[],
    prompt=multimodal_prompt
)

coding_model = ChatOpenAI(model="gpt-4.1")
coding_agent = create_react_agent(
    model=coding_model,
    tools=[code_exec_tool],
    prompt=coding_prompt
)

# right now we implement input -> router -> agent -> output
# next we could make it supervised (so agents can go back ot supervisors) 
# or we could make them talk to one another. And eventually report to supervisor even.

def router(state: MultiState) -> Command[Literal["multimodal_agent", "coding_agent"]]:
    """
    Route to appropriate agent based on media presence (multimodal or coding)
    """
    has_images = bool(state.get("images"))
    has_audios = bool(state.get("audios"))

    if has_images or has_audios:
        goto = "multimodal_agent"
    else: 
        goto = "coding_agent"
    
    return Command(goto=goto)

async def multimodal_agent(state: MultiState) -> Command[Literal[END]]:   # after multimodal -> stop (could change later)
    """
    Handles multimodal inputs with multimodal model
    """

    # construct multimodal input message
    multimodal_msg = prepare_multimodal_message(state)

    # clear history of last message to swap it with the new one
    history = state.get("messages", [])[:-1] if state.get("messages", []) else []
    updated_history = history + multimodal_msg

    result = await multimodal_agent.ainvoke(updated_history)

    return Command(
        update={
            "messages" : result["messages"],
            "images" : [],  # clear images and audios - could change later
            "audios" : []
        },
        goto=END
    )


async def coding_agent(state: MultiState) -> Command[Literal[END]]:
    """
    Handles coding inputs with coding model
    """

    result = await coding_agent.ainvoke(state["messages"])

    return Command(
        update={
            "messages" : result["messages"],
            "images" : [],  # clear images and audios just for safety- could change later
            "audios" : []
        },
        goto=END
    )

def get_builder(checkpointer) -> StateGraph:
    """
    Get the builder for the graph
    """
    builder = StateGraph(MultiState)
    # nodes
    builder.add_node("router", router)
    builder.add_node("multimodal_agent", multimodal_agent)
    builder.add_node("coding_agent", coding_agent)
    # edges
    builder.add_edge(START, "router")
    # no need for conditional edges, router uses Command(goto=...)

    graph = builder.compile(checkpointer=checkpointer)

    # save the graph display to file
    img = graph.get_graph().draw_mermaid_png() # returns bytes
    # save the bytes to file 
    with open("./src/graph.png", "wb") as f:
        f.write(img)
    print("Graph display saved to ./src/graph.png")

    return graph

if __name__ == "__main__":
    checkpointer = InMemorySaver()
    graph = get_builder(checkpointer)


