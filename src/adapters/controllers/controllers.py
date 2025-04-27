
from fastapi import APIRouter, status
from src.adapters.services.services import Service
from src.schemas.schemas import (
    ConversationInput,
    MessageInput,
)
from src.core.domain.model import (
    SenderEnum,
)


router = APIRouter(
    prefix="/conversation",
    tags=["Conversation"],
)

service = Service()

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_conversation(conversation: ConversationInput):
    conversation = await service.create(conversation)
    return conversation

@router.get("/user/{user_id}", status_code=status.HTTP_200_OK)
async def find_conversations_by_user_id(user_id: str):
    conversations = await service.find_by_user_id(user_id)
    return conversations

@router.post("/{conversation_id}/message", status_code=status.HTTP_200_OK)
async def add_message(conversation_id: str, message: MessageInput):
    sender = SenderEnum.user
    messages = await service.add_message(conversation_id, message, sender)
    return messages