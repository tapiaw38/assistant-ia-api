import pytest
from unittest.mock import MagicMock, call
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional

from pymongo.errors import PyMongoError

from src.core.use_cases.conversation.use_case import CreateUseCase
from src.schemas.schemas import ConversationInput, ConversationOutput, ProfileOutput, MessageOutput
from src.core.domain.model import Conversation, Profile # Removed Message, MessageContent as they are not directly instantiated if no initial message is created by this UC

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

def create_sample_conversation_model(
    conversation_id: str = None,
    user_id: str = "user_test_123",
    title: str = "Test Conversation",
    profile: Profile = None,
    messages: Optional[List] = None # Ensure this can be None or empty for the test
) -> Conversation:
    profile_instance = profile or create_sample_profile_model(user_id=user_id)
    # The Conversation model's default for messages (e.g., default_factory=list) will handle it if not provided.
    # For testing, explicitly passing None or [] is clearer.
    return Conversation(
        id=conversation_id or str(uuid4()),
        user_id=user_id,
        title=title,
        profile=profile_instance,
        messages=messages if messages is not None else [], # Or None, if Conversation model handles Optional[List[Message]] without default_factory
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

def create_sample_conversation_input(title: str = "New Conversation Title") -> ConversationInput:
    return ConversationInput(title=title)

# Fixtures
@pytest.fixture
def mock_profile_repository():
    return MagicMock()

@pytest.fixture
def mock_conversation_repository():
    return MagicMock()

@pytest.fixture
def mock_context_factory(mock_profile_repository, mock_conversation_repository):
    factory_instance = MagicMock()
    repositories_mock = MagicMock()
    repositories_mock.profile = mock_profile_repository
    repositories_mock.conversation = mock_conversation_repository
    factory_instance.repositories = repositories_mock

    mock_factory = MagicMock(return_value=factory_instance)
    return mock_factory

@pytest.fixture
def create_use_case(mock_context_factory):
    return CreateUseCase(context_factory=mock_context_factory)

# Test Scenarios for CreateUseCase.execute()

def test_create_use_case_success(
    create_use_case: CreateUseCase,
    mock_profile_repository: MagicMock,
    mock_conversation_repository: MagicMock
):
    # Arrange
    user_id = "user_success_123"
    input_title = "Successful Conversation"
    conversation_input = create_sample_conversation_input(title=input_title)

    sample_profile = create_sample_profile_model(user_id=user_id)
    mock_profile_repository.find_by_user_id.return_value = sample_profile

    created_conversation_id = str(uuid4())
    mock_conversation_repository.create.return_value = created_conversation_id

    # CreateUseCase should not add any messages.
    # The Conversation model returned by find_by_id should have no messages or model's default.
    mocked_found_conversation = create_sample_conversation_model(
        conversation_id=created_conversation_id,
        user_id=user_id,
        title=input_title,
        profile=sample_profile,
        messages=[] # Explicitly empty, or None if model allows and output schema handles it
    )
    mock_conversation_repository.find_by_id.return_value = mocked_found_conversation

    # Act
    result = create_use_case.execute(conversation_input, user_id)

    # Assert
    mock_profile_repository.find_by_user_id.assert_called_once_with(user_id)

    args, kwargs = mock_conversation_repository.create.call_args
    created_conv_arg: Conversation = args[0]
    assert isinstance(created_conv_arg, Conversation)
    assert created_conv_arg.title == input_title
    assert created_conv_arg.user_id == user_id
    assert created_conv_arg.profile is not None
    assert created_conv_arg.profile.id == sample_profile.id
    # Assert that the CreateUseCase does NOT add an initial message.
    # The `messages` field will be what the Conversation model defaults to when not provided.
    # Pydantic models default Optional[List[X]] to None if no default_factory.
    # If Conversation has `messages: List[Message] = Field(default_factory=list)`, it would be [].
    # Explicitly verify the expected default value for the `messages` field.
    assert created_conv_arg.messages is None  # Verify if the default is None
    assert created_conv_arg.messages == []  # Verify if the default is an empty list


    mock_conversation_repository.find_by_id.assert_called_once_with(created_conversation_id)

    assert isinstance(result, ConversationOutput)
    assert result.id == mocked_found_conversation.id
    assert result.title == mocked_found_conversation.title
    assert result.user_id == mocked_found_conversation.user_id
    assert isinstance(result.profile, ProfileOutput)
    assert result.profile.id == sample_profile.id

    # Assert messages based on what find_by_id returned (which should be none/empty)
    if mocked_found_conversation.messages is None:
        assert result.messages is None or result.messages == [] # Depending on ConversationOutput schema (e.g. Optional or default_factory=list)
    else:
        assert result.messages == []


def test_create_use_case_profile_not_found(
    create_use_case: CreateUseCase,
    mock_profile_repository: MagicMock,
    mock_conversation_repository: MagicMock
):
    # Arrange
    user_id = "user_profile_nf_456"
    conversation_input = create_sample_conversation_input()

    mock_profile_repository.find_by_user_id.side_effect = Exception("Profile not found")

    # Act & Assert
    with pytest.raises(Exception, match="Error executing CreateUseCase: Profile not found"):
        create_use_case.execute(conversation_input, user_id)

    mock_conversation_repository.create.assert_not_called()


def test_create_use_case_conversation_repo_create_error(
    create_use_case: CreateUseCase,
    mock_profile_repository: MagicMock,
    mock_conversation_repository: MagicMock
):
    # Arrange
    user_id = "user_conv_create_err_789"
    conversation_input = create_sample_conversation_input()

    sample_profile = create_sample_profile_model(user_id=user_id)
    mock_profile_repository.find_by_user_id.return_value = sample_profile

    mock_conversation_repository.create.side_effect = PyMongoError("DB create error")

    # Act & Assert
    with pytest.raises(Exception, match="Error interacting with database: DB create error"):
        create_use_case.execute(conversation_input, user_id)

def test_create_use_case_conversation_repo_find_by_id_error(
    create_use_case: CreateUseCase,
    mock_profile_repository: MagicMock,
    mock_conversation_repository: MagicMock
):
    # Arrange
    user_id = "user_find_err_012"
    conversation_input = create_sample_conversation_input()

    sample_profile = create_sample_profile_model(user_id=user_id)
    mock_profile_repository.find_by_user_id.return_value = sample_profile

    created_conversation_id = str(uuid4())
    mock_conversation_repository.create.return_value = created_conversation_id

    mock_conversation_repository.find_by_id.side_effect = PyMongoError("DB find error after create")

    # Act & Assert
    with pytest.raises(Exception, match="Error interacting with database: DB find error after create"):
        create_use_case.execute(conversation_input, user_id)


def test_create_use_case_conversation_not_found_after_creation(
    create_use_case: CreateUseCase,
    mock_profile_repository: MagicMock,
    mock_conversation_repository: MagicMock
):
    # Arrange
    user_id = "user_conv_nf_after_create_345"
    conversation_input = create_sample_conversation_input()

    sample_profile = create_sample_profile_model(user_id=user_id)
    mock_profile_repository.find_by_user_id.return_value = sample_profile

    created_conversation_id = str(uuid4())
    mock_conversation_repository.create.return_value = created_conversation_id

    mock_conversation_repository.find_by_id.return_value = None # Conversation not found

    # Act & Assert
    with pytest.raises(Exception, match="Error executing CreateUseCase: Error conversation not found after creation"):
        create_use_case.execute(conversation_input, user_id)

# Removed Message, MessageContent from domain model imports as they are not directly instantiated by this use case test logic.
# ConversationOutput schema should handle messages being None or [] from the model.
# create_sample_conversation_model now creates conversations with messages=[] or messages=None.
# Success test verifies that CreateUseCase does not add messages, and output reflects this.
