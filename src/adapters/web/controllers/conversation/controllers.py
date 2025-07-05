
from fastapi import APIRouter, status, Depends, Request, Query
from src.schemas.schemas import (
    ConversationInput,
    MessageInput,
)
from src.core.domain.model import (
    SenderEnum,
)
from src.adapters.services.services import Services
from typing import Optional


router = APIRouter(
    prefix="/conversation",
    tags=["Conversation"],
)

def get_instance() -> Services:
    return Services.get_instance()

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation: ConversationInput,
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    conversation = await service.conversation.create(conversation, user_id)
    return conversation

@router.get("/user", status_code=status.HTTP_200_OK)
async def find_conversations_by_user_id(
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    conversations = await service.conversation.find_by_user_id(user_id)
    return conversations

@router.post("/{conversation_id}/message", status_code=status.HTTP_200_OK)
async def add_message(
    conversation_id: str,
    message: MessageInput,
    request: Request,
    service: Services = Depends(get_instance),
    has_image_processor: Optional[str] = Query(None, regex="^(activate|deactivate)$")
):
    sender = SenderEnum.user
    user_id = request.state.user.get("user_id")
    
    messages = await service.conversation.add_message(
        conversation_id, 
        message, 
        sender, 
        user_id, 
        has_image_processor
    )
    return messages

@router.delete("/{conversation_id}/message", status_code=status.HTTP_200_OK)
async def delete_all_messages(
    conversation_id: str,
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    messages = await service.conversation.delete_all_messages(conversation_id, user_id)
    return messages