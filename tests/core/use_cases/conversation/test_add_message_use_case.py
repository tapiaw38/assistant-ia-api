import pytest
import pytest_asyncio # For async fixtures if needed, though not strictly used here
from unittest.mock import MagicMock, AsyncMock # AsyncMock for async methods
from uuid import uuid4
from datetime import datetime, timezone

from pymongo.errors import PyMongoError

from src.core.use_cases.conversation.use_case import AddMessageUseCase
from src.schemas.schemas import MessageInput, MessageListOutput, MessageOutputData, ProfileOutput, MessageContentOutput
from src.core.domain.model import Conversation, Profile, Message, SenderEnum, MessageContent

# Helper Functions
def create_sample_profile_model(user_id: str = "user_test_123", profile_id: str = None) -> Profile:
    return Profile(
        id=profile_id or str(uuid4()),
        user_id=user_id,
        name="Test User",
        email="test@example.com",
        picture="http://example.com/pic.jpg",
        api_keys=[],
        files=[],
        iteration_limit=10,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

def create_sample_message_model(
    content: str,
    sender: SenderEnum,
    created_at: datetime = None,
    message_id: str = None
) -> Message:
    return Message(
        id=message_id or str(uuid4()),
        author_id=str(uuid4()), # Generic author_id for simplicity in message model
        content=MessageContent(text=content),
        sender=sender,
        created_at=created_at or datetime.now(timezone.utc)
    )

def create_sample_conversation_model(
    conversation_id: str,
    user_id: str,
    messages: list[Message] = None,
    profile_id: str = None
) -> Conversation:
    profile = create_sample_profile_model(user_id=user_id, profile_id=profile_id)
    return Conversation(
        id=conversation_id,
        user_id=user_id, # This is the conversation's user_id, distinct from message author_id
        title="Test Conversation",
        profile=profile,
        messages=messages if messages is not None else [],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

def create_message_input(content: str, context: Optional[str] = None) -> MessageInput:
    return MessageInput(content=content, context=context)

# Fixtures
@pytest.fixture
def mock_conversation_repository():
    return MagicMock()

@pytest.fixture
def mock_openai_integration():
    # For async methods like 'ask', use AsyncMock
    mock = MagicMock()
    mock.ask = AsyncMock() # Use AsyncMock for the async method 'ask'
    return mock

@pytest.fixture
def mock_context_factory(mock_conversation_repository, mock_openai_integration):
    factory_instance = MagicMock()
    repositories_mock = MagicMock()
    integrations_mock = MagicMock()

    repositories_mock.conversation = mock_conversation_repository
    integrations_mock.openai = mock_openai_integration

    factory_instance.repositories = repositories_mock
    factory_instance.integrations = integrations_mock

    mock_factory = MagicMock(return_value=factory_instance)
    return mock_factory

@pytest.fixture
def add_message_use_case(mock_context_factory):
    return AddMessageUseCase(context_factory=mock_context_factory)

# Test Scenarios for AddMessageUseCase.execute()

@pytest.mark.asyncio
async def test_add_message_success_first_user_message(
    add_message_use_case: AddMessageUseCase,
    mock_conversation_repository: MagicMock,
    mock_openai_integration: MagicMock
):
    # Arrange
    conversation_id = str(uuid4())
    user_id = "user_first_message_123"
    message_input = create_message_input("Hello, Assistant!", context="Test Context")

    # Conversation with no prior messages
    conversation = create_sample_conversation_model(conversation_id, user_id, messages=[])
    mock_conversation_repository.find_by_id.return_value = conversation

    assistant_response_content = "Hello, User! How can I help?"
    mock_openai_integration.ask.return_value = assistant_response_content

    # Act
    result = await add_message_use_case.execute(conversation_id, message_input, SenderEnum.user, user_id)

    # Assert
    mock_conversation_repository.find_by_id.assert_called_once_with(conversation_id)

    # Check call to OpenAI integration
    mock_openai_integration.ask.assert_awaited_once()
    call_args = mock_openai_integration.ask.call_args
    assert call_args[0][0] == message_input.content # query
    assert call_args[0][1] == message_input.context # context
    assert call_args[0][2] == "" # existing_messages_str (empty for first message)
    assert call_args[0][3] == conversation.profile # profile

    # Check call to update_messages_by_id
    mock_conversation_repository.update_messages_by_id.assert_called_once()
    update_call_args = mock_conversation_repository.update_messages_by_id.call_args
    assert update_call_args[0][0] == conversation_id
    updated_messages_list = update_call_args[0][1]
    assert len(updated_messages_list) == 2

    # User's message
    assert updated_messages_list[0].content.text == message_input.content
    assert updated_messages_list[0].sender == SenderEnum.user

    # Assistant's message
    assert updated_messages_list[1].content.text == assistant_response_content
    assert updated_messages_list[1].sender == SenderEnum.assistant

    # Check result
    assert isinstance(result, MessageListOutput)
    assert len(result.data) == 2
    assert result.data[0].content.text == message_input.content
    assert result.data[0].sender == SenderEnum.user.value # Schema uses value
    assert result.data[1].content.text == assistant_response_content
    assert result.data[1].sender == SenderEnum.assistant.value


@pytest.mark.asyncio
async def test_add_message_success_subsequent_user_message(
    add_message_use_case: AddMessageUseCase,
    mock_conversation_repository: MagicMock,
    mock_openai_integration: MagicMock
):
    # Arrange
    conversation_id = str(uuid4())
    user_id = "user_next_message_456"

    existing_user_msg = create_sample_message_model("Old question", SenderEnum.user)
    existing_assistant_msg = create_sample_message_model("Old answer", SenderEnum.assistant)
    initial_messages = [existing_user_msg, existing_assistant_msg]

    conversation = create_sample_conversation_model(conversation_id, user_id, messages=initial_messages)
    mock_conversation_repository.find_by_id.return_value = conversation

    new_message_input = create_message_input("New question", context="New Context")
    new_assistant_response = "New answer"
    mock_openai_integration.ask.return_value = new_assistant_response

    # Act
    result = await add_message_use_case.execute(conversation_id, new_message_input, SenderEnum.user, user_id)

    # Assert
    mock_conversation_repository.find_by_id.assert_called_once_with(conversation_id)

    expected_existing_messages_str = f"{SenderEnum.user.value}: {existing_user_msg.content.text}\n{SenderEnum.assistant.value}: {existing_assistant_msg.content.text}\n"
    mock_openai_integration.ask.assert_awaited_once_with(
        new_message_input.content,
        new_message_input.context,
        expected_existing_messages_str,
        conversation.profile
    )

    update_call_args = mock_conversation_repository.update_messages_by_id.call_args
    assert update_call_args[0][0] == conversation_id
    updated_messages_list = update_call_args[0][1]
    assert len(updated_messages_list) == 4 # 2 existing + 2 new
    assert updated_messages_list[0] == existing_user_msg
    assert updated_messages_list[1] == existing_assistant_msg
    assert updated_messages_list[2].content.text == new_message_input.content
    assert updated_messages_list[3].content.text == new_assistant_response

    assert isinstance(result, MessageListOutput)
    assert len(result.data) == 4

@pytest.mark.asyncio
async def test_add_message_conversation_not_found(add_message_use_case: AddMessageUseCase, mock_conversation_repository: MagicMock):
    # Arrange
    conversation_id = "non_existent_convo"
    user_id = "user_convo_nf_789"
    message_input = create_message_input("Doesn't matter")
    mock_conversation_repository.find_by_id.return_value = None

    # Act & Assert
    with pytest.raises(Exception, match="Error executing AddMessageUseCase: Conversation not found"):
        await add_message_use_case.execute(conversation_id, message_input, SenderEnum.user, user_id)

@pytest.mark.asyncio
async def test_add_message_user_not_authorized(add_message_use_case: AddMessageUseCase, mock_conversation_repository: MagicMock):
    # Arrange
    conversation_id = str(uuid4())
    actual_user_id = "owner_user"
    intruder_user_id = "intruder_user"
    message_input = create_message_input("Trying to snoop")

    conversation = create_sample_conversation_model(conversation_id, actual_user_id) # Belongs to actual_user_id
    mock_conversation_repository.find_by_id.return_value = conversation

    # Act & Assert
    with pytest.raises(Exception, match="Error executing AddMessageUseCase: User not authorized to add message"):
        await add_message_use_case.execute(conversation_id, message_input, SenderEnum.user, intruder_user_id)

@pytest.mark.asyncio
async def test_add_message_openai_integration_error(
    add_message_use_case: AddMessageUseCase,
    mock_conversation_repository: MagicMock,
    mock_openai_integration: MagicMock
):
    # Arrange
    conversation_id = str(uuid4())
    user_id = "user_openai_err_123"
    message_input = create_message_input("Query")

    conversation = create_sample_conversation_model(conversation_id, user_id, messages=[])
    mock_conversation_repository.find_by_id.return_value = conversation

    mock_openai_integration.ask.side_effect = Exception("OpenAI API error")

    # Act & Assert
    with pytest.raises(Exception, match="Error executing AddMessageUseCase: OpenAI API error"):
        await add_message_use_case.execute(conversation_id, message_input, SenderEnum.user, user_id)

    # Ensure update_messages_by_id was not called if OpenAI fails
    mock_conversation_repository.update_messages_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_add_message_repo_update_error(
    add_message_use_case: AddMessageUseCase,
    mock_conversation_repository: MagicMock,
    mock_openai_integration: MagicMock
):
    # Arrange
    conversation_id = str(uuid4())
    user_id = "user_repo_update_err_456"
    message_input = create_message_input("Final query")

    conversation = create_sample_conversation_model(conversation_id, user_id, messages=[])
    mock_conversation_repository.find_by_id.return_value = conversation

    mock_openai_integration.ask.return_value = "Assistant's reply"
    mock_conversation_repository.update_messages_by_id.side_effect = PyMongoError("DB update error")

    # Act & Assert
    with pytest.raises(Exception, match="Error interacting with the database: DB update error"):
        await add_message_use_case.execute(conversation_id, message_input, SenderEnum.user, user_id)
