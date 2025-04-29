
from fastapi import APIRouter, status, Depends
from src.schemas.schemas import (
    ConversationInput,
    MessageInput,
)
from src.core.domain.model import (
    SenderEnum,
)
from src.adapters.services.services import Services


router = APIRouter(
    prefix="/conversation",
    tags=["Conversation"],
)

def get_instance() -> Services:
    return Services.get_instance()

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation: ConversationInput,
    service: Services = Depends(get_instance)
):
    mock_user_id = "mock_user_id" # TODO: replace with user_id in request
    conversation = await service.conversation.create(conversation, mock_user_id)
    return conversation

@router.get("/user", status_code=status.HTTP_200_OK)
async def find_conversations_by_user_id(
    service: Services = Depends(get_instance)
):
    mock_user_id = "mock_user_id"
    conversations = await service.conversation.find_by_user_id(mock_user_id)
    return conversations

@router.post("/{conversation_id}/message", status_code=status.HTTP_200_OK)
async def add_message(
    conversation_id: str,
    message: MessageInput,
    service: Services = Depends(get_instance)
):
    sender = SenderEnum.user
    messages = await service.conversation.add_message(conversation_id, message, sender)
    return messages