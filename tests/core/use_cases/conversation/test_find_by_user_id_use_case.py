import pytest
from unittest.mock import MagicMock
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional

from pymongo.errors import PyMongoError

from src.core.use_cases.conversation.use_case import FindByUserIdUseCase
from src.schemas.schemas import ConversationListOutput, ConversationOutputData, ProfileOutput, MessageOutput # Assuming MessageOutput is used by ConversationOutputData
from src.core.domain.model import Conversation, Profile, Message, MessageContent # MessageContent for Message

# Helper Functions
def create_sample_profile_model(user_id: str = "user_test_123", profile_id: Optional[str] = None) -> Profile:
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

def create_sample_message_model(message_id: Optional[str] = None, author_id: str = "author123", text: str = "Hello") -> Message:
    return Message(
        id=message_id or str(uuid4()),
        author_id=author_id,
        content=MessageContent(text=text),
        created_at=datetime.now(timezone.utc)
    )

def create_sample_conversation_model(
    user_id: str,
    conversation_id: Optional[str] = None,
    title: str = "Test Conversation",
    messages: Optional[List[Message]] = None,
    profile: Optional[Profile] = None
) -> Conversation:
    profile_instance = profile or create_sample_profile_model(user_id=user_id)
    return Conversation(
        id=conversation_id or str(uuid4()),
        user_id=user_id, # Ensure this is set correctly
        title=title,
        profile=profile_instance,
        messages=messages if messages is not None else [],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

# Fixtures
@pytest.fixture
def mock_conversation_repository():
    return MagicMock()

@pytest.fixture
def mock_context_factory(mock_conversation_repository):
    factory_instance = MagicMock()
    repositories_mock = MagicMock()
    repositories_mock.conversation = mock_conversation_repository
    factory_instance.repositories = repositories_mock

    mock_factory = MagicMock(return_value=factory_instance)
    return mock_factory

@pytest.fixture
def find_by_user_id_use_case(mock_context_factory):
    return FindByUserIdUseCase(context_factory=mock_context_factory)

# Test Scenarios for FindByUserIdUseCase.execute()

def test_find_by_user_id_success_conversations_found(
    find_by_user_id_use_case: FindByUserIdUseCase,
    mock_conversation_repository: MagicMock
):
    # Arrange
    user_id = "user123_with_convos"

    # Create a couple of sample conversations for this user
    profile1 = create_sample_profile_model(user_id=user_id, profile_id="profile1")
    messages1 = [create_sample_message_model(text="Message 1 in Convo 1")]
    convo1 = create_sample_conversation_model(
        user_id=user_id,
        conversation_id="convo1",
        title="First Conversation",
        messages=messages1,
        profile=profile1
    )

    profile2 = create_sample_profile_model(user_id=user_id, profile_id="profile2") # Profile can be different per convo if needed by model
    messages2 = [create_sample_message_model(text="Message 1 in Convo 2")]
    convo2 = create_sample_conversation_model(
        user_id=user_id,
        conversation_id="convo2",
        title="Second Conversation",
        messages=messages2,
        profile=profile2
    )

    mock_conversation_repository.find_user_id.return_value = [convo1, convo2]

    # Act
    result = find_by_user_id_use_case.execute(user_id)

    # Assert
    mock_conversation_repository.find_user_id.assert_called_once_with(user_id)

    assert isinstance(result, ConversationListOutput)
    assert len(result.data) == 2

    # Check data for the first conversation
    result_convo1_data = result.data[0]
    assert isinstance(result_convo1_data, ConversationOutputData)
    assert result_convo1_data.id == convo1.id
    assert result_convo1_data.title == convo1.title
    assert result_convo1_data.user_id == convo1.user_id # Check user_id directly on ConversationOutputData
    assert isinstance(result_convo1_data.profile, ProfileOutput)
    assert result_convo1_data.profile.id == profile1.id # convo1.profile.id
    assert result_convo1_data.profile.user_id == user_id
    assert len(result_convo1_data.messages) == 1
    assert result_convo1_data.messages[0].id == messages1[0].id
    assert result_convo1_data.messages[0].content.text == messages1[0].content.text

    # Check data for the second conversation
    result_convo2_data = result.data[1]
    assert isinstance(result_convo2_data, ConversationOutputData)
    assert result_convo2_data.id == convo2.id
    assert result_convo2_data.title == convo2.title
    assert result_convo2_data.user_id == convo2.user_id
    assert isinstance(result_convo2_data.profile, ProfileOutput)
    assert result_convo2_data.profile.id == profile2.id # convo2.profile.id
    assert result_convo2_data.profile.user_id == user_id
    assert len(result_convo2_data.messages) == 1
    assert result_convo2_data.messages[0].id == messages2[0].id
    assert result_convo2_data.messages[0].content.text == messages2[0].content.text


def test_find_by_user_id_success_no_conversations_found(
    find_by_user_id_use_case: FindByUserIdUseCase,
    mock_conversation_repository: MagicMock
):
    # Arrange
    user_id = "user456_no_convos"
    mock_conversation_repository.find_user_id.return_value = []

    # Act
    result = find_by_user_id_use_case.execute(user_id)

    # Assert
    mock_conversation_repository.find_user_id.assert_called_once_with(user_id)
    assert isinstance(result, ConversationListOutput)
    assert len(result.data) == 0


def test_find_by_user_id_conversation_repository_error(
    find_by_user_id_use_case: FindByUserIdUseCase,
    mock_conversation_repository: MagicMock
):
    # Arrange
    user_id = "user789_db_error"
    mock_conversation_repository.find_user_id.side_effect = PyMongoError("DB error")

    # Act & Assert
    with pytest.raises(Exception, match="Error interacting with database: DB error"):
        find_by_user_id_use_case.execute(user_id)
