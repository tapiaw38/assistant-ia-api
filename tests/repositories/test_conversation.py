import pytest
from unittest.mock import MagicMock
from datetime import datetime
from uuid import uuid4
from src.adapters.repositories.repositories import Repository
from src.core.domain.model import Conversation


@pytest.fixture
def mock_client():
    return MagicMock()

@pytest.fixture
def repository(mock_client):
    return Repository(client=mock_client)

def build_conversation(**kwargs):
    defaults = {
        "id": str(uuid4()), 
        "user_id": "123",
        "title": "Test Conversation",
        "created_at": datetime.now()
    }
    defaults.update(kwargs)
    return Conversation(**defaults)

@pytest.mark.parametrize("name,test_case", [
    (
        "when conversation creation succeeds",
        {
            "input": lambda: build_conversation(),
            "mock_behavior": lambda client, conv: setattr(
                client.insert_one.return_value, "inserted_id", conv.id
            ),
            "expected_result": lambda conv: conv.id,
            "expect_error": None,
            "test_fn": "create"
        },
    ),
    (
        "when conversation creation fails",
        {
            "input": lambda: build_conversation(),
            "mock_behavior": lambda client, conv: setattr(
                client.insert_one, "side_effect", 
                Exception("Error inserting conversation"),
            ),
            "expected_result": None,
            "expect_error": Exception,
            "test_fn": "create"
        },
    ),
])

def test_conversation_cases(repository, mock_client, name, test_case):
    conversation = test_case["input"]()
    test_fn = test_case["test_fn"]

    test_case["mock_behavior"](mock_client, conversation)

    if test_case["expect_error"]:
        with pytest.raises(test_case["expect_error"]):
            getattr(repository, test_fn)(conversation.id if test_fn != "create" else conversation)
    else:
        result = getattr(repository, test_fn)(conversation.id if test_fn != "create" else conversation)
        expected = test_case["expected_result"](conversation)

        if isinstance(expected, Conversation):
            assert result.dict() == expected.dict()
        else:
            assert result == expected
