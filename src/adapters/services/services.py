from fastapi import HTTPException, status
from src.core.use_cases.use_cases import (
    CreateUseCase,
    FindByUserIdUseCase,
    AddMessageUseCase,
)
from src.adapters.repositories.repositories import Repository
from src.schemas.schemas import (
    ConversationInput,
    MessageInput,
)
from src.core.domain.model import (
    SenderEnum,
)
from src.core.platform.nosql.mongo_db import get_db

client = next(get_db())

class Service:
    def __init__(self, client=client):
        self.conversation_repository = Repository(client)

    async def create(self, conversation: ConversationInput):
        try:
            usecase = CreateUseCase(self.conversation_repository)
            created_conversation = usecase.execute(conversation)

            if created_conversation is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Conversation not created")

            return created_conversation

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def find_by_user_id(self, user_id: str):
        try:
            usecase = FindByUserIdUseCase(self.conversation_repository)
            conversations = usecase.execute(user_id)
            return conversations
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def add_message(self, conversation_id: str, message: MessageInput, sender: SenderEnum):
        try:
            usecase = AddMessageUseCase(self.conversation_repository)
            messages = await usecase.execute(conversation_id, message, sender)
            return messages
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))