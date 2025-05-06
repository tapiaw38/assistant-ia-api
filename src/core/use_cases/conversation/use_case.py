from src.schemas.schemas import (
    ConversationInput,
    MessageInput,
    ConversationOutput,
    ConversationListOutput,
    MessageListOutput,
)
from src.core.domain.model import (
    Conversation,
    Message,
    SenderEnum,
    Profile,
)
from datetime import datetime, timezone
from pymongo.errors import PyMongoError
from src.core.platform.appcontext.appcontext import Factory
from uuid import uuid4

class CreateUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, conversation: ConversationInput, user_id: str) -> Conversation:
        try:
            profile = self.context_factory.repositories.profile.find_by_user_id(user_id)
            generated_id = str(uuid4())
            new_conversation = Conversation(
                _id= generated_id,
                profile= Profile(
                    id=profile.id,
                    user_id=profile.user_id,
                    assistant_name=profile.assistant_name,
                    business_name=profile.business_name,
                    functions=profile.functions,
                    business_context=profile.business_context,
                    created_at=profile.created_at,
                    updated_at=profile.updated_at,
                ),
                title=conversation.title,
                created_at=datetime.now(timezone.utc),
            )

            conversation_id = self.context_factory.repositories.conversation.create(new_conversation)

            created_conversation =  self.context_factory.repositories.conversation.find_by_id(conversation_id)

            if not created_conversation:
                raise Exception("Error conversation not found after creation")

            return ConversationOutput.from_output(created_conversation)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing CreateUseCase: {e}")

class FindByUserIdUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, user_id: str):
        try:
            conversations = self.context_factory.repositories.conversation.find_user_id(user_id)
            return ConversationListOutput.from_output(conversations)
        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing FindByUserIdUseCase: {e}")

class AddMessageUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def convert_messages_to_string(self, messages: list[Message]) -> str:
        message_objects = [
            f"{message.sender}: {message.content}"
            for message in messages
        ]

        return " - ".join(message_objects)

    async def execute(self, conversation_id: str, message: MessageInput, sender: SenderEnum, user_id: str) -> list[Message]:
        new_message = Message(
            content=message.content,
            sender=sender,
            created_at=datetime.now(timezone.utc),
        )

        try:
            search_conversation = self.context_factory.repositories.conversation.find_by_id(conversation_id)
            if not search_conversation:
                raise Exception("Conversation not found")

            if user_id != search_conversation.profile.user_id:
                raise Exception("User not authorized to add message")

            if not hasattr(search_conversation, "messages") or search_conversation.messages is None:
                search_conversation.messages = []

            if any(msg.id == new_message.id for msg in search_conversation.messages):
                raise Exception("Message with the same ID already exists in the conversation")

            search_conversation.messages.append(new_message)

            response = await self.context_factory.integrations.openai.ask(
                new_message.content,
                message.context,
                self.convert_messages_to_string(search_conversation.messages),
                search_conversation.profile,
            )

            response_message = Message(
                content=response,
                sender="assistant",
                created_at=datetime.now(timezone.utc),
            )

            search_conversation.messages.append(response_message)

            self.context_factory.repositories.conversation.update_messages_by_id(
                conversation_id, 
                search_conversation.messages,
            )

            return MessageListOutput.from_output(search_conversation.messages)

        except PyMongoError as e:
            raise Exception(f"Error interacting with the database: {e}")

        except Exception as e:
            raise Exception(f"Error executing AddMessageUseCase: {e}")


class DeleteAllMessagesUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    async def execute(self, conversation_id: str, user_id: str):
        try:
            search_conversation = self.context_factory.repositories.conversation.find_by_id(conversation_id)
            if not search_conversation:
                raise Exception("Conversation not found")

            if user_id != search_conversation.profile.user_id:
                raise Exception("User not authorized to delete messages")

            self.context_factory.repositories.conversation.update_messages_by_id(
                conversation_id,
                [],
            )

            search_conversation.messages = []

            return MessageListOutput.from_output(search_conversation.messages)

        except PyMongoError as e:
            raise Exception(f"Error interacting with the database: {e}")

        except Exception as e:
            raise Exception(f"Error executing DeleteAllMessagesUseCase: {e}")