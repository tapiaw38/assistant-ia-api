from pydantic import BaseModel, Field
from typing import Optional, List
from src.core.domain.model import (
    Profile,
    Conversation,  
    Message,
    ApiKey,
)
from datetime import datetime


class ConversationInput(BaseModel):
    title: str


class MessageInput(BaseModel):
    content: str
    context: Optional[str]


class ProfileInput(BaseModel):
    assistant_name: Optional[str] = None
    business_name: Optional[str] = None
    functions: Optional[str] = None
    business_context: Optional[str] = None


class ProfileStatusInput(BaseModel):
    is_active: bool


class ProfileOutputData(BaseModel):
    id: str
    user_id: str
    assistant_name: str
    business_name: str
    functions: str
    business_context: str
    is_active: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class ProfileOutput(BaseModel):
    data: ProfileOutputData

    @staticmethod
    def from_output(profile: Profile):
        return ProfileOutput(
            data=ProfileOutputData(
                id=profile.id,
                user_id=profile.user_id,
                assistant_name=profile.assistant_name,
                business_name=profile.business_name,
                functions=profile.functions,
                business_context=profile.business_context,
                is_active=profile.is_active,
                created_at=profile.created_at,
                updated_at=profile.updated_at,
            )
        )


class ConversationOutputData(BaseModel):
    id: Optional[str]
    title: str
    created_at: datetime
    profile: Profile
    messages:  Optional[List[Message]]


class ConversationOutput(BaseModel):
    data: ConversationOutputData

    @staticmethod
    def from_output(conversation: Conversation):
        return ConversationOutput(
            data=ConversationOutputData(
                id=conversation.id,
                title=conversation.title,
                created_at=conversation.created_at,
                profile=conversation.profile,
                messages=conversation.messages,
            )
        )


class ConversationListOutput(BaseModel):
    data: List[ConversationOutputData]

    @staticmethod
    def from_output(conversations: list[Conversation]):
        return ConversationListOutput(
            data=[
                ConversationOutput.from_output(conversation).data
                for conversation in conversations
            ]
        )


class MessageOutputData(BaseModel):
    id: str
    content: str
    sender: str
    created_at: datetime


class MessageOutput(BaseModel):
    data: MessageOutputData

    @staticmethod
    def from_output(message: Message):
        return MessageOutput(
            data=MessageOutputData(
                id=message.id,
                content=message.content,
                sender=message.sender,
                created_at=message.created_at,
            )
        )

class MessageListOutput(BaseModel):
    data: List[MessageOutputData]

    @staticmethod
    def from_output(messages: list[Message]):
        return MessageListOutput(
            data=[
                MessageOutput.from_output(message).data
                for message in messages
            ]
        )


class ApiKeyInput(BaseModel):
    description: str
    limit: int


class ApiKeyOutputData(BaseModel):
    id: str
    value: str
    description: str
    limit: int
    is_active: bool
    created_at: datetime


class ApiKeyOutput(BaseModel):
    data: ApiKeyOutputData

    @staticmethod
    def from_output(api_key: ApiKey):
        return ApiKeyOutput(
            data=ApiKeyOutputData(
                id=api_key.id,
                value=api_key.value,
                description=api_key.description,
                limit=api_key.limit,
                is_active=api_key.is_active,
                created_at=api_key.created_at,
            )
        )

class ApiKeyListOutput(BaseModel):
    data: List[ApiKeyOutputData]

    @staticmethod
    def from_output(api_keys: list[ApiKey]):
        return ApiKeyListOutput(
            data=[
                ApiKeyOutput.from_output(api_key).data
                for api_key in api_keys
            ]
        )