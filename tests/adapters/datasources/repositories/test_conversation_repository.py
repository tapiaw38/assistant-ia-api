import pytest
from unittest.mock import MagicMock
from uuid import uuid4
from datetime import datetime
from typing import List, Optional

from pymongo.errors import PyMongoError

from src.adapters.datasources.repositories.conversation.repository import Repository
from src.core.domain.model import Conversation, Profile, Message, MessageContent

# Helper functions to create sample data
def create_sample_profile(user_id: str = "user_123", name: str = "Test User") -> Profile:
    return Profile(user_id=user_id, name=name, email=f"{name.replace(' ', '').lower()}@example.com", picture="http://example.com/pic.jpg")

def create_sample_message(
    id: str = str(uuid4()),
    author_id: str = "author_123",
    content_text: str = "Hello!",
    created_at: Optional[datetime] = None
) -> Message:
    return Message(
        id=id,
        author_id=author_id,
        content=MessageContent(text=content_text),
        created_at=created_at or datetime.now()
    )

def create_sample_conversation(
    id: str = str(uuid4()),
    user_id: str = "user_123",
    title: str = "Test Conversation",
    profile: Optional[Profile] = None,
    messages: Optional[List[Message]] = None,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None
) -> Conversation:
    return Conversation(
        id=id,
        user_id=user_id,
        title=title,
        profile=profile or create_sample_profile(user_id=user_id),
        messages=messages if messages is not None else [create_sample_message()],
        created_at=created_at or datetime.now(),
        updated_at=updated_at or datetime.now()
    )

@pytest.fixture
def mock_datasources():
    datasources = MagicMock()
    datasources.no_sql_conversations_client = MagicMock()
    return datasources

@pytest.fixture
def conversation_repository(mock_datasources):
    return Repository(datasources=mock_datasources)

# Tests for create method
def test_create_success(conversation_repository: Repository, mock_datasources: MagicMock):
    sample_conv = create_sample_conversation()
    mock_insert_result = MagicMock()
    mock_insert_result.inserted_id = sample_conv.id
    mock_datasources.no_sql_conversations_client.insert_one.return_value = mock_insert_result

    result_id = conversation_repository.create(sample_conv)

    assert result_id == sample_conv.id
    mock_datasources.no_sql_conversations_client.insert_one.assert_called_once_with(sample_conv.dict(by_alias=True))

def test_create_pymongo_error(conversation_repository: Repository, mock_datasources: MagicMock):
    sample_conv = create_sample_conversation()
    mock_datasources.no_sql_conversations_client.insert_one.side_effect = PyMongoError("DB error")

    with pytest.raises(Exception) as excinfo:
        conversation_repository.create(sample_conv)
    assert "Error creating conversation" in str(excinfo.value)

# Tests for find_by_id method
def test_find_by_id_success_found(conversation_repository: Repository, mock_datasources: MagicMock):
    conv_id = "test_conv_id_1"
    sample_conv_data = create_sample_conversation(id=conv_id).dict(by_alias=True)
    # MongoDB stores id as _id
    sample_conv_data["_id"] = sample_conv_data.pop("id")
    mock_datasources.no_sql_conversations_client.find_one.return_value = sample_conv_data

    result_conv = conversation_repository.find_by_id(conv_id)

    assert result_conv is not None
    assert result_conv.id == conv_id
    mock_datasources.no_sql_conversations_client.find_one.assert_called_once_with({"_id": conv_id})

def test_find_by_id_success_not_found(conversation_repository: Repository, mock_datasources: MagicMock):
    conv_id = "test_conv_id_not_found"
    mock_datasources.no_sql_conversations_client.find_one.return_value = None

    result_conv = conversation_repository.find_by_id(conv_id)

    assert result_conv is None
    mock_datasources.no_sql_conversations_client.find_one.assert_called_once_with({"_id": conv_id})

def test_find_by_id_pymongo_error(conversation_repository: Repository, mock_datasources: MagicMock):
    conv_id = "test_conv_id_err"
    mock_datasources.no_sql_conversations_client.find_one.side_effect = PyMongoError("DB error")

    with pytest.raises(Exception) as excinfo:
        conversation_repository.find_by_id(conv_id)
    assert f"Error finding conversation by id {conv_id}" in str(excinfo.value)


# Tests for find_user_id method
def test_find_user_id_success_found(conversation_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_test_1"
    sample_conv_data_1 = create_sample_conversation(user_id=user_id, id="conv1").dict(by_alias=True)
    sample_conv_data_1["_id"] = sample_conv_data_1.pop("id")
    sample_conv_data_2 = create_sample_conversation(user_id=user_id, id="conv2").dict(by_alias=True)
    sample_conv_data_2["_id"] = sample_conv_data_2.pop("id")

    mock_datasources.no_sql_conversations_client.find.return_value = [sample_conv_data_1, sample_conv_data_2]

    result_convs = conversation_repository.find_user_id(user_id)

    assert len(result_convs) == 2
    assert all(isinstance(c, Conversation) for c in result_convs)
    assert result_convs[0].id == "conv1"
    assert result_convs[1].id == "conv2"
    mock_datasources.no_sql_conversations_client.find.assert_called_once_with({"profile.user_id": user_id})

def test_find_user_id_success_not_found(conversation_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_test_not_found"
    mock_datasources.no_sql_conversations_client.find.return_value = []

    result_convs = conversation_repository.find_user_id(user_id)

    assert len(result_convs) == 0
    mock_datasources.no_sql_conversations_client.find.assert_called_once_with({"profile.user_id": user_id})

def test_find_user_id_pymongo_error(conversation_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_test_err"
    mock_datasources.no_sql_conversations_client.find.side_effect = PyMongoError("DB error")

    with pytest.raises(Exception) as excinfo:
        conversation_repository.find_user_id(user_id)
    assert f"Error finding conversations by user id {user_id}" in str(excinfo.value)


# Tests for update_profile_by_user_id method
def test_update_profile_by_user_id_success(conversation_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_for_update"
    sample_prof = create_sample_profile(user_id=user_id, name="Updated Name")

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_datasources.no_sql_conversations_client.update_many.return_value = mock_update_result

    result_user_id = conversation_repository.update_profile_by_user_id(user_id, sample_prof)

    assert result_user_id == user_id
    expected_update_doc = {"$set": {"profile": sample_prof.dict()}}
    mock_datasources.no_sql_conversations_client.update_many.assert_called_once_with(
        {"profile.user_id": user_id},
        expected_update_doc
    )

def test_update_profile_by_user_id_not_found(conversation_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_not_found_for_update"
    sample_prof = create_sample_profile(user_id=user_id)

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 0
    mock_datasources.no_sql_conversations_client.update_many.return_value = mock_update_result

    with pytest.raises(Exception) as excinfo:
        conversation_repository.update_profile_by_user_id(user_id, sample_prof)
    assert f"Profile not found for user ID {user_id}" in str(excinfo.value)


def test_update_profile_by_user_id_pymongo_error(conversation_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_update_err"
    sample_prof = create_sample_profile(user_id=user_id)
    mock_datasources.no_sql_conversations_client.update_many.side_effect = PyMongoError("DB error")

    with pytest.raises(Exception) as excinfo:
        conversation_repository.update_profile_by_user_id(user_id, sample_prof)
    assert f"Error updating profile for user ID {user_id}" in str(excinfo.value)


# Tests for update_messages_by_id method
def test_update_messages_by_id_success(conversation_repository: Repository, mock_datasources: MagicMock):
    conv_id = "conv_for_msg_update"
    sample_messages = [create_sample_message(content_text="Updated message")]

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_datasources.no_sql_conversations_client.update_one.return_value = mock_update_result

    conversation_repository.update_messages_by_id(conv_id, sample_messages)

    expected_messages_dict = [msg.dict() for msg in sample_messages]
    expected_update_doc = {"$set": {"messages": expected_messages_dict}}
    mock_datasources.no_sql_conversations_client.update_one.assert_called_once_with(
        {"_id": conv_id},
        expected_update_doc
    )

def test_update_messages_by_id_not_found(conversation_repository: Repository, mock_datasources: MagicMock):
    conv_id = "conv_not_found_for_msg_update"
    sample_messages = [create_sample_message()]

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 0
    mock_datasources.no_sql_conversations_client.update_one.return_value = mock_update_result

    with pytest.raises(Exception) as excinfo:
        conversation_repository.update_messages_by_id(conv_id, sample_messages)
    assert f"Conversation not found with ID {conv_id} to update messages" in str(excinfo.value)


def test_update_messages_by_id_pymongo_error(conversation_repository: Repository, mock_datasources: MagicMock):
    conv_id = "conv_msg_update_err"
    sample_messages = [create_sample_message()]
    mock_datasources.no_sql_conversations_client.update_one.side_effect = PyMongoError("DB error")

    with pytest.raises(Exception) as excinfo:
        conversation_repository.update_messages_by_id(conv_id, sample_messages)
    assert f"Error updating messages for conversation ID {conv_id}" in str(excinfo.value)

# Ensure all necessary imports are present
from src.core.domain.model import MessageContent # Already imported Message, Profile, Conversation
from pymongo.errors import PyMongoError # Already imported
# import uuid # Not strictly needed as uuid4 is used directly
from datetime import datetime # Already imported
from typing import List, Optional # Already imported
