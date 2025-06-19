from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime
from uuid import uuid4


class SenderEnum(str, Enum):
    user = "user"
    assistant = "assistant"


class ApiKey(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    user_id: str
    value: str
    description: str
    is_active: bool
    limit: int
    created_at: datetime


class File(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    name: str
    url: str
    created_at: datetime

    class Config:
        allow_population_by_field_name = True

class Integration(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    name: str
    type: str
    config: dict
    created_at: datetime
    updated_at: datetime

    class Config:
        allow_population_by_field_name = True


class Profile(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    assistant_name: Optional[str] = None
    business_name: Optional[str] = None
    functions: Optional[str] = None
    business_context: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
    is_active: Optional[bool] = Field(default=True)
    api_keys: Optional[List[ApiKey]] = Field(default=None)
    iteration_limit: Optional[int] = Field(default=None)
    files: Optional[List[File]] = Field(default=None)
    integrations: Optional[List[Integration]] = Field(default=None)

    class Config:
        allow_population_by_field_name = True


class Message(BaseModel):
    id: str = Field(
        alias="_id",
        default_factory=lambda: str(uuid4()),
    )
    content: str
    sender: SenderEnum
    created_at: datetime


class Conversation(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    title: str
    created_at: datetime
    profile: Profile
    messages: Optional[List[Message]] = Field(default=None)

    class Config:
        allow_population_by_field_name = True