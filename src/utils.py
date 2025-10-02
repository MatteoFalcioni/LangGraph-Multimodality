from langchain_core.messages import HumanMessage
from state import MultiState

def prepare_multimodal_message(state: MultiState) -> HumanMessage:
    """
    Helper to create multimodal message from state

    Returns:
        HumanMessage: The multimodal message
    """
    messages = state.get("messages", [])
    
    # Get text from last message or use default
    if messages and isinstance(messages[-1].content, str):
        text = messages[-1].content
    else:
        text = "Analyze this media"
    
    content = [{"type": "text", "text": text}]
    
    # Add images
    for img_b64 in state.get("images", []):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    
    # Add audios
    for audio_b64 in state.get("audios", []):
        content.append({
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": "wav"}
        })
    
    return HumanMessage(content=content)