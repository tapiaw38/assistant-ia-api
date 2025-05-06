from fastapi import HTTPException, status
from src.schemas.schemas import (
    ConversationInput,
    MessageInput,
)
from src.core.domain.model import (
    SenderEnum,
)
from src.core.use_cases.use_cases import Conversation


class ConversationService:
    def __init__(self, usecase: Conversation):
        self.usecase = usecase

    async def create(self, conversation: ConversationInput, user_id: str):
        try:
            created_conversation = self.usecase.create_usecase.execute(conversation, user_id)

            if created_conversation is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Conversation not created")

            return created_conversation

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def find_by_user_id(self, user_id: str):
        try:
            conversations = self.usecase.find_by_user_id_usecase.execute(user_id)
            return conversations
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def add_message(self, conversation_id: str, message: MessageInput, sender: SenderEnum, user_id: str):
        try:
            messages = await self.usecase.add_message_usecase.execute(conversation_id, message, sender, user_id)
            return messages
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def delete_all_messages(self, conversation_id: str, user_id: str):
        try:
            messages = await self.usecase.delete_all_messages_usecase.execute(conversation_id, user_id)
            return messages
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))