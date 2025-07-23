
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
    prefix="/mercadolibre",
    tags=["MercadoLibre"],
)

def get_instance() -> Services:
    return Services.get_instance()

@router.post("/auth", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation: ConversationInput,
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    conversation = await service.conversation.create(conversation, user_id)
    return conversation

@router.post("/questions", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation: ConversationInput,
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    conversation = await service.conversation.create(conversation, user_id)
    return conversation

@router.get("/response_all", status_code=status.HTTP_200_OK)
async def find_conversations_by_user_id(
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    conversations = await service.conversation.find_by_user_id(user_id)
    return conversations