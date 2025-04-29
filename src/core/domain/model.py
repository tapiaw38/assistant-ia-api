from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime
from uuid import uuid4


class SenderEnum(str, Enum):
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        alias="_id",
    )
    content: str
    sender: SenderEnum
    created_at: datetime


class Conversation(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        alias="_id",
    )
    title: str
    created_at: datetime
    user_id: str
    messages: Optional[List[Message]] = Field(default_factory=list)

    class Config:
        allow_population_by_field_name = True


class ApiKey(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        alias="_id",
    )
    value: str
    description: str
    created_at: datetime


class Profile(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        alias="_id",
    )

    assistant_name: Optional[str] = None
    business_name: Optional[str] = None
    prompt: Optional[str] = None
    prompt_context: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
    api_keys: list[ApiKey] = Field(default_factory=list)

    class Config:
        allow_population_by_field_name = True