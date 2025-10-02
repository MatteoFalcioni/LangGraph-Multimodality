from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import List, Union
from typing_extensions import Annotated

def add_b64(left : Union[List[str], None], right : Union[List[str], None]) -> List:
    """
    Reducer to combine two lists of base64 strings.
    If right is empty list, it clears (allows nodes to reset media).
    In the future: could swap lists with dicitonaries with boolean flags for already inserted in chat.
    """

    if right is not None and len(right) == 0:
        return []   # explicit clear, when passing `update: "images" : []` or `"audios" : []`

    if left is None:  # init left list
        left = []
    
    if right is None:   # init right list
        right = []
    
    return left + right # (l, r) -> (l+r)


class MultiState(AgentState):
    """State of multimodal agent, inherits from AgentState, so it gets `messages` and `remaining_steps` keys for free"""
    images : Annotated[List[str], add_b64]
    audios : Annotated[List[str], add_b64]