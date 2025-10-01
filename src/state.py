from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import List, Union
from typing_extensions import Annotated

def add_b64(left : Union[List[str], None], right : Union[List[str], None]) -> List:
    """
    Reducer to combine two lists of base64 strings
    """
    if left is None:
        return []
    if right is None:
        return []
    
    return left + right


class MultiState(AgentState):
    """State of multimodal agent, inherits from AgentState, so it gets `messages` and `remaining_steps` keys for free"""
    images : Annotated[List[str], add_b64]
    audios : Annotated[List[str], add_b64]