from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from src.core.domain.model import (
    Conversation,
    Message,
    Profile,
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


class ProfileInput(BaseModel):
    user_id: Optional[str] = None
    assistant_name: Optional[str] = None
    business_name: Optional[str] = None
    prompt: Optional[str] = None
    prompt_context: Optional[str] = None

class ProfileOutput(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    assistant_name: str
    business_name: str
    prompt: str
    prompt_context: str
    created_at: datetime

    @classmethod
    def from_profile(cls, profile: Profile):
        return cls(
            _id=profile.id,
            user_id=profile.user_id,
            assistant_name=profile.assistant_name,
            business_name=profile.business_name,
            prompt=profile.prompt,
            prompt_context=profile.prompt_context,
            created_at=profile.created_at,
        )

    def to_profile(self) -> Profile:
        return Profile(
            id=self.id,
            user_id=self.user_id,
            assistant_name=self.assistant_name,
            business_name=self.business_name,
            prompt=self.prompt,
            prompt_context=self.prompt_context,
            created_at=self.created_at,
        )