from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

from state import MultiState
from sys_prompt import PROMPT

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

multimodal_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=PROMPT
)


def prepare_input(state: MultiState):
    """
    Prepare the input for the model
    """

    # problem: here we are appedning images if there are any in state, and the first time this is fine, 
    # but at the second pass we already have the first image, we keep appending it to chat history
    # we should make a system that if there is a *new* image (audio) appends it to the last message in chat history. 
    # how to do that? maybe with a dict of images/audios that contains a boolean flag 'in_memory'? Would work, but seems really mechanical
    # also, what if the last message is empty? like we only input an image or an audio without text. Then we would be appending an img/audio 
    # to the last message in chat history, which could be totally unrelated

    # the new thing of such system is that we could have input with no messages. I never had to deal with that before. 
    # we could have input_state = {"messages" : [], "images" : [<b64>], "audios" : []} for example. 
    # and if no text, we should probably craft a text for the model like look at this image or listen to this audio.

    # can we then just check the last state in order to do so? 
    
    # let's try to construct an ad-hoc state to pass to the model

    last_state = state[-1]  # Typed Dict w/ messages, images, audios (all lists) and remaining_steps

    last_msg_content = last_state["messages"][-1].get("content", None)
    if last_msg_content is None:
        last_msg_content = "look at this image or listen to this audio"

    input_messages = state.get("messages", [])
    input_messages.append(HumanMessage(content=last_msg_content))
    
    # prepare last msg content list
    content = [
        {
            "type" : "text",
            "text" : last_msg_content
        }
    ]
    
    # extract images and audio lists from **last** state
    img_list = last_state.get("images", [])
    audio_list = last_state.get("audios", [])

    if len(img_list) > 0:
        img_b64 = img_list[-1]
        content.append({
            "type" : "image",
            "image" : img_b64,
            "mime_type" : "image/jpeg"
        })
    if len(audio_list) > 0:
        audio_b64 = audio_list[-1]
        content.append({
            "type" : "audio",
            "audio" : audio_b64,
            "mime_type" : "audio/wav"   # or appropriate mime type
        })

    # construct input message
    prepared_msg = {
        "role" : "user",
        "content" : content
    }

    # im confused, langgraph philosophy is to get state as input and pass state as output
    # and nodes should always return state updates. Im not returning an update
    # Im trying to overwrite the existing state into this one
    # my idea was to modify the last state to be this one, and then in call_llm just use messages as input 

    # is this actually correct? could be tho... we are using state to make an update to messages according to our needs
    # then we call the llm on messages. So imgs and audios list in state are used only to keep track of this stuff
    # so if we make them a dict with a flag that tells us if already inserted in chat or not, then it should work
    # or actually we could even erase them from state after we used them to construct the message...

    # i guess the only doubt i have is with langgraph memory. if we use a checkpointer the graph is called on the state
    # and serializes all the stuff that we pass through. but we will be invoking only in the node call_llm, so it should serialize only our well constructed msg.
    # in the sense that in main we will do `await graph.ainvoke(init, config={"configurable" : {"thread_id": 1}})`
    # but this will just call ainvoke in `call_model`

    return {"messages" : prepared_msg}  # we should return a state update like this. 

async def call_model(state: MultiState):
    """
    Invokes the model (async) 
    """

    result = await multimodal_agent.ainvoke(state["messages"])  # invoking only on messages? 

    last = result["messages"][-1]
    update = AIMessage(content=last.content)
    return {"messages": [update]}   # remember to wrap in list