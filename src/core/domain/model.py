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


