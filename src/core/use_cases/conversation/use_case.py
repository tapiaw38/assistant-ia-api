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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IMAGE_CONFIDENCE_THRESHOLD = 0.3
MAX_IMAGES_TO_SHOW = 3
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
                profile=Profile(**profile.dict()),
                title=conversation.title,
                created_at=datetime.now(timezone.utc),
            )

            conversation_id = self.context_factory.repositories.conversation.create(
                new_conversation
            )

            created_conversation = self.context_factory.repositories.conversation.find_by_id(conversation_id)

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
        return " - ".join([f"{msg.sender}: {msg.content}" for msg in messages])

    async def _process_mercadolibre_integration(self, integration, profile):
        try:
            ml_config = integration.config
            access_token = ml_config.get("access_token")
            ml_user_id = ml_config.get("user_id")

            logger.info(f"MercadoLibre integration - Access token: {access_token[:20]}...")
            logger.info(f"MercadoLibre integration - User ID: {ml_user_id}")

            if not access_token or not ml_user_id:
                logger.warning("MercadoLibre integration missing credentials")
                return

            self.context_factory.integrations.initialize_mercadolibre(access_token, ml_user_id)
            ml_integration = self.context_factory.integrations.mercadolibre

            if not ml_integration:
                logger.error("Failed to initialize MercadoLibre integration")
                return

            async with ml_integration:
                logger.info("Processing MercadoLibre integration...")
                logger.info("Getting user items from MercadoLibre...")
                items = await ml_integration.get_user_items(limit=10)
                logger.info(f"Found {len(items)} user items")

                if items:
                    for item in items[:3]:
                        logger.info(f"- {item.title} ({item.price} {item.currency_id}) - Status: {item.status}")

                    auto_answer = True

                    logger.info("Processing MercadoLibre unanswered questions...")
                    results = await ml_integration.auto_answer_questions_with_ai(
                        self.context_factory.integrations.openai,
                        profile,
                        auto_send=auto_answer
                    )

                    logger.info(f"MercadoLibre processing results: {results['total_questions']} questions, "
                                f"{results['answered']} answered, {results['failed']} failed")

                    if not auto_answer and results['responses']:
                        for response in results['responses'][:3]:
                            logger.info(f"Q: {response['question_text'][:100]}...")
                            logger.info(f"A: {response['suggested_answer'][:200]}...")

                    if auto_answer and results['answered'] > 0:
                        for response in results['responses']:
                            if response['answered']:
                                logger.info(f"✅ Answered Q{response['question_id']}: {response['question_text'][:50]}...")

                    logger.info(f"MercadoLibre integration successful - {len(items)} items, {results['total_questions']} questions processed")
                else:
                    logger.warning("Could not access user items - check token permissions or user has no active items")

        except Exception as e:
            logger.error(f"Error processing MercadoLibre integration: {e}")

    async def execute(self, conversation_id: str, message: MessageInput, sender: SenderEnum, user_id: str, has_image_processor: Optional[str] = None) -> list[Message]:
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

            file_images = []
            formatted_images_list = []
            image_search_keyword = new_message.content

            if has_image_processor == "activate":
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

                max_results_for_search = min(MAX_IMAGES_TO_SHOW, 2) if ENABLE_FAST_IMAGE_PROCESSING else MAX_IMAGES_TO_SHOW
                file_images = await self.context_factory.integrations.openai.search_images_in_files(
                    image_search_keyword,
                    search_conversation.profile,
                    max_results_for_search,
                )
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
                conversation_id, search_conversation.messages
            )

            for integration in search_conversation.profile.integrations:
                if integration.name == "meli":
                    await self._process_mercadolibre_integration(integration, search_conversation.profile)

            if has_image_processor == "activate" and formatted_images_list:
                formatted_images = "\n".join(formatted_images_list)
                if formatted_images:
                    response_with_images = response + f"\n\n{formatted_images}"
                    search_conversation.messages[-1].content = response_with_images

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
            search_conversation = self.context_factory.repositories.conversation.find_by_id(conversation_id)
            if not search_conversation:
                raise Exception("Conversation not found")

            if user_id != search_conversation.profile.user_id:
                raise Exception("User not authorized to delete messages")

            self.context_factory.repositories.conversation.update_messages_by_id(conversation_id, [])
            search_conversation.messages = []

            return MessageListOutput.from_output(search_conversation.messages)

        except PyMongoError as e:
            raise Exception(f"Error interacting with the database: {e}")

        except Exception as e:
            raise Exception(f"Error executing DeleteAllMessagesUseCase: {e}")
