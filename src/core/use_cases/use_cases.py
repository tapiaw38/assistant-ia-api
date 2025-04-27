from src.adapters.repositories.repositories import RepositoryInterface
from src.schemas.schemas import (
    ConversationInput,
    MessageInput,
)
from src.core.domain.model import (
    Conversation,
    Message,
    SenderEnum,
)
from datetime import datetime, timezone
from pymongo.errors import PyMongoError
from src.core.platform.config.service import (
    init_config,
    get_config_service,
)
from src.adapters.integrations.integrations import (
    create_integrations,
)


init_config()
integration = create_integrations(get_config_service())

config = init_config()
integration = create_integrations(get_config_service)

class CreateUseCase:
    def __init__(self, conversation_repository: RepositoryInterface):
        self.conversation_repository = conversation_repository

    def execute(self, conversation: ConversationInput):
        try:
            new_conversation = Conversation(
                user_id=conversation.user_id,
                title=conversation.title,
                created_at=datetime.now(timezone.utc),
            )

            conversation_id = self.conversation_repository.create(new_conversation)

            created_conversation = self.conversation_repository.find_by_id(conversation_id)

            if not created_conversation:
                raise Exception("Error conversation not found after creation")

            return created_conversation

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing CreateUseCase: {e}")

class FindByUserIdUseCase:
    def __init__(self, conversation_repository: RepositoryInterface):
        self.conversation_repository = conversation_repository

    def execute(self, user_id: str):
        try:
            conversations = self.conversation_repository.find_user_id(user_id)
            return conversations
        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing FindByUserIdUseCase: {e}")

class AddMessageUseCase:
    def __init__(self, conversation_repository: RepositoryInterface):
        self.conversation_repository = conversation_repository

    def convert_messages_to_string(self, messages: list[Message]) -> str:
        message_objects = [
            f"{message.sender}: {message.content}"
            for message in messages
        ]

        return " - ".join(message_objects)

    async def execute(self, conversation_id: str, message: MessageInput, sender: SenderEnum) -> list[Message]:
        new_message = Message(
            content=message.content,
            sender=sender,
            created_at=datetime.now(timezone.utc),
        )

        try:
            search_conversation = self.conversation_repository.find_by_id(conversation_id)
            if not search_conversation:
                raise Exception("Conversation not found")

            if not hasattr(search_conversation, "messages") or search_conversation.messages is None:
                search_conversation.messages = []

            if any(msg.id == new_message.id for msg in search_conversation.messages):
                raise Exception("Message with the same ID already exists in the conversation")

            search_conversation.messages.append(new_message)

            response = await integration.openai.ask(
                new_message.content,
                message.context,
                self.convert_messages_to_string(search_conversation.messages)
            )

            response_message = Message(
                content=response,
                sender="assistant",
                created_at=datetime.now(timezone.utc),
            )

            search_conversation.messages.append(response_message)

            self.conversation_repository.update_messages_by_id(
                conversation_id, 
                search_conversation.messages,
            )

            return search_conversation.messages

        except PyMongoError as e:
            raise Exception(f"Error interacting with the database: {e}")

        except Exception as e:
            raise Exception(f"Error executing AddMessageUseCase: {e}")