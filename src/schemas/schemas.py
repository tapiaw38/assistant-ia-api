from pydantic import BaseModel, Field
from typing import Optional, List
from src.core.domain.model import (
    Profile,
    Conversation,  
    Message,
    ApiKey,
)
from datetime import datetime
from fastapi import UploadFile


class ConversationInput(BaseModel):
    title: str


class MessageInput(BaseModel):
    content: str
    context: Optional[str]


class ApiKeyInput(BaseModel):
    description: str


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


class ApiKeyDeleteOutputData(BaseModel):
    id: str


class ApiKeyDeleteOutput(BaseModel):
    data: ApiKeyDeleteOutputData

    @staticmethod
    def from_output(api_key_id: str):
        return ApiKeyDeleteOutput(
            data=ApiKeyDeleteOutputData(
                id=api_key_id,
            )
        )


class FileInput:
    def __init__(self, file: UploadFile):
        if not file:
            raise ValueError("File object cannot be None")
        if not hasattr(file, 'filename'):
            raise ValueError("UploadFile object missing 'filename' attribute")

        if not hasattr(file, 'file'):
            raise ValueError("UploadFile object missing 'file' attribute")

        self.filename = file.filename
        self.file = file.file
        self.file_header = getattr(file, 'headers', {})

        try:
            if hasattr(file, 'size') and file.size is not None:
                self.filesize = file.size
                print(f"FileInput init - size from file.size: {self.filesize}")
            elif self.file and hasattr(self.file, '_file') and hasattr(self.file._file, 'tell'):
                current_pos = self.file._file.tell()
                self.file._file.seek(0, 2)  # Ir al final
                self.filesize = self.file._file.tell()
                self.file._file.seek(current_pos)  # Volver a la posición original
                print(f"FileInput init - size from file._file.tell(): {self.filesize}")
            else:
                self.filesize = 0
                print("FileInput init - size defaulted to 0")
        except (AttributeError, OSError, TypeError) as e:
            print(f"FileInput init - error getting size: {e}")
            self.filesize = 0

        if not self.file:
            raise ValueError(f"File content is None for file: {self.filename}")

        print(f"FileInput init completed - filename: {self.filename}, size: {self.filesize}")

    def __repr__(self):
        return f"FileInput(filename='{self.filename}', filesize={self.filesize}, file_type={type(self.file)})"


class FileOutputData(BaseModel):
    id: str
    name: str
    url: str

class FileOutput(BaseModel):
    data: FileOutputData

    @staticmethod
    def from_output(file: FileOutputData):
        return FileOutput(
            data=FileOutputData(
                id=file.id,
                name=file.name,
                url=file.url,
            )
        )


class FileListOutput(BaseModel):
    data: List[FileOutputData]

    @staticmethod
    def from_output(files: list[FileOutputData]):
        return FileListOutput(
            data=[
                FileOutput.from_output(file).data
                for file in files
            ]
        )


class FileDeleteOutputData(BaseModel):
    id: str


class FileDeleteOutput(BaseModel):
    data: FileDeleteOutputData

    @staticmethod
    def from_output(file_id: str):
        return FileDeleteOutput(
            data=FileDeleteOutputData(
                id=file_id,
            )
        )


class IntegrationInput(BaseModel):
    name: str
    type: str
    config: dict


class IntegrationOutputData(BaseModel):
    id: str
    name: str
    type: str
    config: dict
    created_at: datetime
    updated_at: Optional[datetime] = None


class IntegrationOutput(BaseModel):
    data: IntegrationOutputData

    @staticmethod
    def from_output(integration: IntegrationOutputData):
        return IntegrationOutput(
            data=IntegrationOutputData(
                id=integration.id,
                name=integration.name,
                type=integration.type,
                config=integration.config,
                created_at=integration.created_at,
                updated_at=integration.updated_at if integration.updated_at else None,
            )
        )


class IntegrationListOutput(BaseModel):
    data: List[IntegrationOutputData]

    @staticmethod
    def from_output(integrations: list[IntegrationOutputData]):
        return IntegrationListOutput(
            data=[
                IntegrationOutput.from_output(integration).data
                for integration in integrations
            ]
        )


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
    iteration_limit: Optional[int]
    api_keys: Optional[List[ApiKeyOutputData]]
    files: Optional[List[FileOutputData]]
    integrations: Optional[List[IntegrationOutputData]]


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
                iteration_limit=profile.iteration_limit,
                api_keys=[ApiKeyOutput.from_output(api_key).data for api_key in profile.api_keys] if profile.api_keys else None,
                files=[FileOutput.from_output(file).data for file in profile.files] if profile.files else None,
                integrations=[IntegrationOutput.from_output(integration).data for integration in profile.integrations] if profile.integrations else None,
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