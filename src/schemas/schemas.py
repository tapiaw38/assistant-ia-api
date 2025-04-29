from pydantic import BaseModel
from typing import Optional


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
