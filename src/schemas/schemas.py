from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from src.core.domain.model import (
    Conversation,
    Message,
)


class ConversationInput(BaseModel):
    user_id: str
    title: str

class MessageInput(BaseModel):
    content: str
    context: Optional[str]

class ConversationOutput(BaseModel):
    id: str = Field(alias="_id") 
    user_id: str
    title: str
    created_at: datetime
    messages: List[Message]

    @classmethod
    def from_conversation(cls, conversation: Conversation):
        return cls(
            _id=conversation.id,  
            user_id=conversation.user_id,
            title=conversation.title,
            created_at=conversation.created_at,
        )

    def to_conversation(self) -> Conversation:
        return Conversation(
            id=self.id,
            user_id=self.user_id,
            title=self.title,
            created_at=self.created_at,
        )