import logging
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
from src.adapters.web.utils.image_formatter import format_images_for_chat_response
from uuid import uuid4
from typing import Optional


# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Remove os import and environment variable usage
# Set module-level constants
IMAGE_CONFIDENCE_THRESHOLD = 0.3
MAX_IMAGES_TO_SHOW = 3
# Optimization: Use fast mode to reduce API calls
ENABLE_FAST_IMAGE_PROCESSING = True


class CreateUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, conversation: ConversationInput, user_id: str) -> Conversation:
        try:
            profile = self.context_factory.repositories.profile.find_by_user_id(user_id)
            generated_id = str(uuid4())
            new_conversation = Conversation(
                _id=generated_id,
                profile=Profile(
                    _id=profile.id,
                    user_id=profile.user_id,
                    assistant_name=profile.assistant_name,
                    business_name=profile.business_name,
                    functions=profile.functions,
                    business_context=profile.business_context,
                    files=profile.files,
                    created_at=profile.created_at,
                    updated_at=profile.updated_at,
                ),
                title=conversation.title,
                created_at=datetime.now(timezone.utc),
            )

            conversation_id = self.context_factory.repositories.conversation.create(
                new_conversation
            )

            created_conversation = (
                self.context_factory.repositories.conversation.find_by_id(
                    conversation_id
                )
            )

            if not created_conversation:
                raise Exception("Error: conversation not found after creation")

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
            conversations = self.context_factory.repositories.conversation.find_user_id(
                user_id
            )
            return ConversationListOutput.from_output(conversations)
        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing FindByUserIdUseCase: {e}")

class FindByIdUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, conversation_id: str) -> Optional[Conversation]:
        try:
            conversation = self.context_factory.repositories.conversation.find_by_id(
                conversation_id
            )
            if not conversation:
                raise Exception("Conversation not found")
            return ConversationOutput.from_output(conversation)
        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing FindByIdUseCase: {e}")

class AddMessageUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def convert_messages_to_string(self, messages: list[Message]) -> str:
        """
        Converts a list of Message objects to a single string for context.
        """
        message_objects = [
            f"{message.sender}: {message.content}" for message in messages
        ]
        return " - ".join(message_objects)

    async def execute(
        self,
        conversation_id: str,
        message: MessageInput,
        sender: SenderEnum,
        user_id: str,
        has_image_processor: Optional[str] = None,
    ) -> list[Message]:
        """
        Adds a message to a conversation and processes image extraction if enabled.
        If image processing is activated, first asks OpenAI for the keyword to search in images,
        then uses that keyword for image search instead of the original message content.
        """
        new_message = Message(
            content=message.content,
            sender=sender,
            created_at=datetime.now(timezone.utc),
        )

        try:
            search_conversation = (
                self.context_factory.repositories.conversation.find_by_id(
                    conversation_id
                )
            )
            if not search_conversation:
                logger.error("Conversation not found")
                raise Exception("Conversation not found")

            if user_id != search_conversation.profile.user_id:
                logger.error("User not authorized to add message")
                raise Exception("User not authorized to add message")

            if (
                not hasattr(search_conversation, "messages")
                or search_conversation.messages is None
            ):
                search_conversation.messages = []

            if any(msg.id == new_message.id for msg in search_conversation.messages):
                logger.error("Message with the same ID already exists in the conversation")
                raise Exception(
                    "Message with the same ID already exists in the conversation"
                )

            search_conversation.messages.append(new_message)

            response = await self.context_factory.integrations.openai.ask(
                new_message.content,
                message.context,
                self.convert_messages_to_string(search_conversation.messages),
                search_conversation.profile,
            )

            file_images = []
            formatted_images_list = []
            image_search_keyword = new_message.content

            if has_image_processor == "activate":
                # Ask OpenAI for the keyword to search in images
                logger.info("Requesting image search keyword from OpenAI...")
                keyword_prompt = (
                    "Given the following user message, what is the best single keyword or phrase to use to search for relevant images in the user's files? "
                    "Respond ONLY with the keyword or phrase, no explanation.\n\n"
                    f"User message: {new_message.content}"
                )
                image_search_keyword = await self.context_factory.integrations.openai.ask(
                    keyword_prompt,
                    message.context,
                    self.convert_messages_to_string(search_conversation.messages),
                    search_conversation.profile,
                )
                logger.info(f"Image search keyword received: '{image_search_keyword}'")

                # Use optimized image search with reduced API calls
                if ENABLE_FAST_IMAGE_PROCESSING:
                    logger.info("Using optimized fast image processing mode")
                    # Limit to fewer results to reduce processing time
                    max_results_for_search = min(MAX_IMAGES_TO_SHOW, 2)
                else:
                    max_results_for_search = MAX_IMAGES_TO_SHOW

                file_images = await self.context_factory.integrations.openai.search_images_in_files(
                    image_search_keyword,
                    search_conversation.profile,
                    max_results_for_search,
                )

                logger.info(f"File images found: {len(file_images)}")

                # Filter by confidence threshold
                file_images = [img for img in file_images if getattr(img, 'confidence', 1.0) >= IMAGE_CONFIDENCE_THRESHOLD]

                if file_images:
                    formatted_images = format_images_for_chat_response(
                        file_images, image_search_keyword, max_images_to_show=MAX_IMAGES_TO_SHOW
                    )
                    formatted_images_list.append(formatted_images)

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

            response_with_images = response
            if has_image_processor == "activate" and formatted_images_list:
                formatted_images = "\n".join(formatted_images_list)
                if formatted_images:
                    response_with_images += f"\n\n{formatted_images}"
                    response_message.content = response_with_images
                    search_conversation.messages[-1] = response_message

            return MessageListOutput.from_output(search_conversation.messages)

        except PyMongoError as e:
            logger.error(f"Error interacting with the database: {e}")
            raise Exception(f"Error interacting with the database: {e}")

        except Exception as e:
            logger.error(f"Error executing AddMessageUseCase: {e}")
            raise Exception(f"Error executing AddMessageUseCase: {e}")


class DeleteAllMessagesUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    async def execute(self, conversation_id: str, user_id: str):
        try:
            search_conversation = (
                self.context_factory.repositories.conversation.find_by_id(
                    conversation_id
                )
            )
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
