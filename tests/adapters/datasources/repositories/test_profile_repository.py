import pytest
from unittest.mock import MagicMock, call
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional

from pymongo.errors import PyMongoError, DuplicateKeyError

from src.adapters.datasources.repositories.profile.repository import Repository
from src.core.domain.model import Profile, ApiKey, File

# Helper functions to create sample data
def create_sample_api_key(
    key_id: Optional[str] = None,
    value: Optional[str] = None,
    name: str = "Test Key",
    is_active: bool = True,
    created_at: Optional[datetime] = None,
    last_used_at: Optional[datetime] = None
) -> ApiKey:
    return ApiKey(
        id=key_id or str(uuid4()),
        value=value or str(uuid4().hex), # Generate a hex string for value
        name=name,
        is_active=is_active,
        created_at=created_at or datetime.now(timezone.utc),
        last_used_at=last_used_at
    )

def create_sample_file(
    file_id: Optional[str] = None,
    name: str = "test_file.txt",
    path: str = "/path/to/test_file.txt",
    size_bytes: int = 1024,
    created_at: Optional[datetime] = None
) -> File:
    return File(
        id=file_id or str(uuid4()),
        name=name,
        path=path,
        size_bytes=size_bytes,
        created_at=created_at or datetime.now(timezone.utc)
    )

def create_sample_profile(
    profile_id: Optional[str] = None,
    user_id: str = "user_123",
    name: str = "Test User",
    email: str = "test@example.com",
    picture: Optional[str] = "http://example.com/pic.jpg",
    api_keys: Optional[List[ApiKey]] = None,
    files: Optional[List[File]] = None,
    iteration_limit: int = 100,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None
) -> Profile:
    return Profile(
        id=profile_id or str(uuid4()),
        user_id=user_id,
        name=name,
        email=email,
        picture=picture,
        api_keys=api_keys if api_keys is not None else [create_sample_api_key()],
        files=files if files is not None else [create_sample_file()],
        iteration_limit=iteration_limit,
        created_at=created_at or datetime.now(timezone.utc),
        updated_at=updated_at or datetime.now(timezone.utc)
    )

@pytest.fixture
def mock_datasources():
    datasources = MagicMock()
    datasources.no_sql_profiles_client = MagicMock()
    # Mock the collection itself to mock create_index
    datasources.no_sql_profiles_client.create_index = MagicMock()
    return datasources

@pytest.fixture
def profile_repository(mock_datasources):
    # The __init__ method of Repository calls create_index.
    # So, the mock_datasources.no_sql_profiles_client must already have create_index mocked.
    return Repository(datasources=mock_datasources)

# Test __init__
def test_repository_init(mock_datasources):
    # Repository initialization is handled by the fixture.
    # We just need to assert that create_index was called.
    # The fixture `profile_repository` will trigger the __init__
    Repository(datasources=mock_datasources)
    mock_datasources.no_sql_profiles_client.create_index.assert_called_once_with("user_id", unique=True)

# Tests for create method
def test_create_profile_success(profile_repository: Repository, mock_datasources: MagicMock):
    sample_prof = create_sample_profile()
    mock_insert_result = MagicMock()
    mock_insert_result.inserted_id = sample_prof.id
    mock_datasources.no_sql_profiles_client.insert_one.return_value = mock_insert_result

    result_id = profile_repository.create(sample_prof)

    assert result_id == sample_prof.id
    # Pydantic models by default don't include fields that are None when converting to dict
    # The Profile model needs to be checked for how it serializes.
    # Assuming default Pydantic behavior: exclude_none=True might be relevant if used.
    # For this test, let's assume all fields are present or have defaults.
    profile_dict = sample_prof.dict(by_alias=True)
    mock_datasources.no_sql_profiles_client.insert_one.assert_called_once_with(profile_dict)

def test_create_profile_duplicate_key_error(profile_repository: Repository, mock_datasources: MagicMock):
    sample_prof = create_sample_profile()
    mock_datasources.no_sql_profiles_client.insert_one.side_effect = DuplicateKeyError("Duplicate key")

    with pytest.raises(DuplicateKeyError):
        profile_repository.create(sample_prof)

def test_create_profile_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    sample_prof = create_sample_profile()
    mock_datasources.no_sql_profiles_client.insert_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo: # Repository wraps PyMongoError in RuntimeError
        profile_repository.create(sample_prof)
    assert "Failed to create profile" in str(excinfo.value)


# Tests for find_by_id method
def test_find_by_id_success_found(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "test_profile_id_1"
    # Ensure _id is used for mongo documents
    sample_profile_data = create_sample_profile(profile_id=profile_id).dict(by_alias=True)
    db_doc = sample_profile_data.copy()
    db_doc["_id"] = db_doc.pop("id") # MongoDB uses _id

    mock_datasources.no_sql_profiles_client.find_one.return_value = db_doc

    result_profile = profile_repository.find_by_id(profile_id)

    assert result_profile is not None
    assert result_profile.id == profile_id
    mock_datasources.no_sql_profiles_client.find_one.assert_called_once_with({"_id": profile_id})

def test_find_by_id_not_found(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "test_profile_id_not_found"
    mock_datasources.no_sql_profiles_client.find_one.return_value = None

    with pytest.raises(Exception) as excinfo:
        profile_repository.find_by_id(profile_id)
    assert f"Profile not found with ID: {profile_id}" in str(excinfo.value)


def test_find_by_id_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "test_profile_id_err"
    mock_datasources.no_sql_profiles_client.find_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.find_by_id(profile_id)
    assert f"Error finding profile by ID {profile_id}" in str(excinfo.value)


# Tests for find_by_user_id method
def test_find_by_user_id_success_found(profile_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_test_1"
    sample_profile_data = create_sample_profile(user_id=user_id).dict(by_alias=True)
    db_doc = sample_profile_data.copy()
    db_doc["_id"] = db_doc.pop("id")


    mock_datasources.no_sql_profiles_client.find_one.return_value = db_doc
    result_profile = profile_repository.find_by_user_id(user_id)

    assert result_profile is not None
    assert result_profile.user_id == user_id
    mock_datasources.no_sql_profiles_client.find_one.assert_called_once_with({"user_id": user_id})


def test_find_by_user_id_not_found(profile_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_test_not_found"
    mock_datasources.no_sql_profiles_client.find_one.return_value = None

    with pytest.raises(Exception) as excinfo:
        profile_repository.find_by_user_id(user_id)
    assert f"Profile not found for user_id: {user_id}" in str(excinfo.value)

def test_find_by_user_id_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    user_id = "user_test_err"
    mock_datasources.no_sql_profiles_client.find_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.find_by_user_id(user_id)
    assert f"Error finding profile by user_id {user_id}" in str(excinfo.value)

# Tests for find_all_user_ids method
def test_find_all_user_ids_success_populated(profile_repository: Repository, mock_datasources: MagicMock):
    db_docs = [
        {"user_id": "user1", "_id": "id1"},
        {"user_id": "user2", "_id": "id2"}
    ]
    mock_datasources.no_sql_profiles_client.find.return_value = db_docs

    result_ids = profile_repository.find_all_user_ids()

    assert result_ids == ["user1", "user2"]
    mock_datasources.no_sql_profiles_client.find.assert_called_once_with({}, ["user_id"])

def test_find_all_user_ids_success_empty(profile_repository: Repository, mock_datasources: MagicMock):
    mock_datasources.no_sql_profiles_client.find.return_value = []

    result_ids = profile_repository.find_all_user_ids()

    assert result_ids == []
    mock_datasources.no_sql_profiles_client.find.assert_called_once_with({}, ["user_id"])

def test_find_all_user_ids_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    mock_datasources.no_sql_profiles_client.find.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.find_all_user_ids()
    assert "Failed to retrieve all user IDs" in str(excinfo.value)

# Tests for update method
def test_update_profile_success(profile_repository: Repository, mock_datasources: MagicMock):
    sample_prof = create_sample_profile(name="Updated Name")
    profile_id = sample_prof.id

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    updated_profile = profile_repository.update(sample_prof)

    assert updated_profile.id == profile_id
    assert updated_profile.name == "Updated Name"

    profile_dict = sample_prof.dict(by_alias=True)
    profile_dict.pop("id", None) # id is not updated in the set command
    expected_update_doc = {"$set": profile_dict}
    mock_datasources.no_sql_profiles_client.update_one.assert_called_once_with(
        {"_id": profile_id},
        expected_update_doc
    )

def test_update_profile_not_found(profile_repository: Repository, mock_datasources: MagicMock):
    sample_prof = create_sample_profile()

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 0
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    with pytest.raises(Exception) as excinfo:
        profile_repository.update(sample_prof)
    assert f"Profile not found with ID: {sample_prof.id} for update" in str(excinfo.value)

def test_update_profile_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    sample_prof = create_sample_profile()
    mock_datasources.no_sql_profiles_client.update_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.update(sample_prof)
    assert f"Error updating profile with ID {sample_prof.id}" in str(excinfo.value)


# Tests for update_api_keys method
def test_update_api_keys_success(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_with_keys"
    api_keys = [create_sample_api_key(name="NewKey")]

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    profile_repository.update_api_keys(profile_id, api_keys)

    api_keys_dict = [key.dict() for key in api_keys]
    expected_update_doc = {"$set": {"api_keys": api_keys_dict, "updated_at": mock_datasources.no_sql_profiles_client.find_one_and_update.return_value.updated_at}} # Need to mock time or capture arg

    # Using call_args to check parts of the update document due to datetime.now()
    args, kwargs = mock_datasources.no_sql_profiles_client.update_one.call_args
    assert args[0] == {"_id": profile_id}
    assert "$set" in args[1]
    assert args[1]["$set"]["api_keys"] == api_keys_dict
    assert "updated_at" in args[1]["$set"]


def test_update_api_keys_not_found(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_keys_not_found"
    api_keys = [create_sample_api_key()]

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 0
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    with pytest.raises(Exception) as excinfo:
        profile_repository.update_api_keys(profile_id, api_keys)
    assert f"Profile not found with ID: {profile_id} to update API keys" in str(excinfo.value)


def test_update_api_keys_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_keys_err"
    api_keys = [create_sample_api_key()]
    mock_datasources.no_sql_profiles_client.update_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.update_api_keys(profile_id, api_keys)
    assert f"Error updating API keys for profile ID {profile_id}" in str(excinfo.value)


# Tests for deactivate_api_key method
def test_deactivate_api_key_success(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_deactivate_key"
    key_id = "key_to_deactivate"

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_update_result.modified_count = 1 # Ensure a key was actually modified
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    profile_repository.deactivate_api_key(profile_id, key_id)

    args, kwargs = mock_datasources.no_sql_profiles_client.update_one.call_args
    assert args[0] == {"_id": profile_id, "api_keys.id": key_id, "api_keys.is_active": True}
    assert "$set" in args[1]
    assert args[1]["$set"]["api_keys.$.is_active"] == False
    assert "api_keys.$.last_used_at" in args[1]["$set"] # Assuming last_used_at is updated
    assert "updated_at" in args[1]["$set"]

def test_deactivate_api_key_not_found_or_inactive(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_deactivate_key_not_found"
    key_id = "key_not_found_or_inactive"

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 0 # Or modified_count = 0 if profile found but key not/inactive
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    with pytest.raises(Exception) as excinfo:
        profile_repository.deactivate_api_key(profile_id, key_id)
    assert f"API key {key_id} not found, not active, or profile {profile_id} not found" in str(excinfo.value)

def test_deactivate_api_key_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_deactivate_err"
    key_id = "key_err"
    mock_datasources.no_sql_profiles_client.update_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.deactivate_api_key(profile_id, key_id)
    assert f"Error deactivating API key {key_id} for profile {profile_id}" in str(excinfo.value)


# Tests for find_by_api_key_value method
def test_find_by_api_key_value_success(profile_repository: Repository, mock_datasources: MagicMock):
    api_key_value = "secret_key_value"
    sample_prof = create_sample_profile()
    db_doc = sample_prof.dict(by_alias=True)
    db_doc["_id"] = db_doc.pop("id")

    mock_datasources.no_sql_profiles_client.find_one.return_value = db_doc

    result_profile = profile_repository.find_by_api_key_value(api_key_value)

    assert result_profile is not None
    assert result_profile.id == sample_prof.id
    mock_datasources.no_sql_profiles_client.find_one.assert_called_once_with(
        {"api_keys": {"$elemMatch": {"value": api_key_value, "is_active": True}}}
    )

def test_find_by_api_key_value_not_found(profile_repository: Repository, mock_datasources: MagicMock):
    api_key_value = "non_existent_key"
    mock_datasources.no_sql_profiles_client.find_one.return_value = None

    with pytest.raises(Exception) as excinfo:
        profile_repository.find_by_api_key_value(api_key_value)
    assert f"Profile not found for API key value: {api_key_value}" in str(excinfo.value)


def test_find_by_api_key_value_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    api_key_value = "key_find_err"
    mock_datasources.no_sql_profiles_client.find_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.find_by_api_key_value(api_key_value)
    assert f"Error finding profile by API key value {api_key_value}" in str(excinfo.value)

# Tests for update_iteration_limit method
def test_update_iteration_limit_success(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_limit_update"
    new_limit = 200

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    profile_repository.update_iteration_limit(profile_id, new_limit)

    args, kwargs = mock_datasources.no_sql_profiles_client.update_one.call_args
    assert args[0] == {"_id": profile_id}
    assert "$set" in args[1]
    assert args[1]["$set"]["iteration_limit"] == new_limit
    assert "updated_at" in args[1]["$set"]

def test_update_iteration_limit_not_found(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_limit_not_found"
    new_limit = 200

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 0
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    with pytest.raises(Exception) as excinfo:
        profile_repository.update_iteration_limit(profile_id, new_limit)
    assert f"Profile not found with ID: {profile_id} to update iteration limit" in str(excinfo.value)

def test_update_iteration_limit_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_limit_err"
    new_limit = 200
    mock_datasources.no_sql_profiles_client.update_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.update_iteration_limit(profile_id, new_limit)
    assert f"Error updating iteration limit for profile ID {profile_id}" in str(excinfo.value)

# Tests for update_files method
def test_update_files_success(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_files_update"
    files = [create_sample_file(name="new_file.doc")]

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    profile_repository.update_files(profile_id, files)

    files_dict = [f.dict() for f in files]
    args, kwargs = mock_datasources.no_sql_profiles_client.update_one.call_args
    assert args[0] == {"_id": profile_id}
    assert "$set" in args[1]
    assert args[1]["$set"]["files"] == files_dict
    assert "updated_at" in args[1]["$set"]


def test_update_files_not_found(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_files_not_found"
    files = [create_sample_file()]

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 0
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    with pytest.raises(Exception) as excinfo:
        profile_repository.update_files(profile_id, files)
    assert f"Profile not found with ID: {profile_id} to update files" in str(excinfo.value)

def test_update_files_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_files_err"
    files = [create_sample_file()]
    mock_datasources.no_sql_profiles_client.update_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.update_files(profile_id, files)
    assert f"Error updating files for profile ID {profile_id}" in str(excinfo.value)


# Tests for delete_file_by_id method
def test_delete_file_by_id_success(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_delete_file"
    file_id_to_delete = "file_to_delete_id"

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1
    mock_update_result.modified_count = 1 # Important: check that a file was actually removed
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    profile_repository.delete_file_by_id(profile_id, file_id_to_delete)

    args, kwargs = mock_datasources.no_sql_profiles_client.update_one.call_args
    assert args[0] == {"_id": profile_id}
    assert "$pull" in args[1]
    assert args[1]["$pull"] == {"files": {"id": file_id_to_delete}}
    assert "$set" in args[1] # For updated_at
    assert "updated_at" in args[1]["$set"]

def test_delete_file_by_id_not_found_or_file_missing(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_delete_file_profile_nf"
    file_id_to_delete = "file_nf_id"

    mock_update_result = MagicMock()
    mock_update_result.matched_count = 1 # Profile might be found
    mock_update_result.modified_count = 0 # But file not found, so nothing modified
    mock_datasources.no_sql_profiles_client.update_one.return_value = mock_update_result

    with pytest.raises(Exception) as excinfo:
        profile_repository.delete_file_by_id(profile_id, file_id_to_delete)
    assert f"File with ID {file_id_to_delete} not found in profile {profile_id}, or profile not found." in str(excinfo.value)

    # Test case where profile itself is not found
    mock_update_result.matched_count = 0
    with pytest.raises(Exception) as excinfo:
        profile_repository.delete_file_by_id(profile_id, file_id_to_delete)
    assert f"File with ID {file_id_to_delete} not found in profile {profile_id}, or profile not found." in str(excinfo.value)


def test_delete_file_by_id_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_delete_file_err"
    file_id_to_delete = "file_err_id"
    mock_datasources.no_sql_profiles_client.update_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.delete_file_by_id(profile_id, file_id_to_delete)
    assert f"Error deleting file {file_id_to_delete} from profile {profile_id}" in str(excinfo.value)


# Tests for get_file_by_id method
def test_get_file_by_id_success(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_get_file"
    file_to_find = create_sample_file(name="found_file.txt")

    # The repository method queries for the profile and then extracts the file in Python.
    # So, mock find_one to return the profile containing the file.
    profile_doc = create_sample_profile(profile_id=profile_id, files=[file_to_find]).dict(by_alias=True)
    db_doc = profile_doc.copy()
    db_doc["_id"] = db_doc.pop("id")
    mock_datasources.no_sql_profiles_client.find_one.return_value = db_doc

    found_file = profile_repository.get_file_by_id(profile_id, file_to_find.id)

    assert found_file is not None
    assert found_file.id == file_to_find.id
    assert found_file.name == "found_file.txt"
    # The query should be for the profile, then filtering the file.
    # The actual mongo query used by the method is {"_id": profile_id, "files.id": file_id}
    # but the method as written fetches the whole profile if only profile_id is used
    # and then iterates. Let's test the actual implementation.
    # The method in repository.py uses: self.collection.find_one({"_id": profile_id, "files.id": file_id}, {"files.$": 1})
    mock_datasources.no_sql_profiles_client.find_one.assert_called_once_with(
        {"_id": profile_id, "files.id": file_to_find.id},
        {"files.$": 1}
    )

def test_get_file_by_id_profile_not_found(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_get_file_profile_nf"
    file_id = "file_id_get"
    mock_datasources.no_sql_profiles_client.find_one.return_value = None # Profile not found

    with pytest.raises(Exception) as excinfo:
        profile_repository.get_file_by_id(profile_id, file_id)
    assert f"Profile not found with ID: {profile_id} when trying to get file {file_id}" in str(excinfo.value)


def test_get_file_by_id_file_not_found_in_profile(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_get_file_file_nf"
    file_id_to_find = "file_id_not_in_profile"

    # Profile found, but the files array (as returned by projection) is empty or doesn't contain the file.
    # This happens if the $elemMatch (or files.$ in this case) doesn't find the file.
    profile_doc_no_matching_file = create_sample_profile(profile_id=profile_id, files=[]).dict(by_alias=True)
    # Or, the projection returns a document but the 'files' array is empty because no element matched.
    # Example: find_one returns {"_id": profile_id} but not "files" or "files": []
    mock_datasources.no_sql_profiles_client.find_one.return_value = {"_id": profile_id} # No 'files' field or empty 'files'

    with pytest.raises(Exception) as excinfo:
        profile_repository.get_file_by_id(profile_id, file_id_to_find)
    assert f"File with ID {file_id_to_find} not found in profile {profile_id}" in str(excinfo.value)

    # Case: Profile found, files array exists, but specific file not in it (covered by the same logic if projection works as expected)
    # If find_one returns a document like { "_id": "profile_id", "files": [some_other_file_dict] }
    # The code `profile_data.get("files", [])[0]` would either fail if files is missing/empty
    # or return the wrong file if the projection isn't specific.
    # The projection {"files.$": 1} ensures only the matched file (or none) is returned in the array.
    # So if the find_one returns a doc, but files array is empty, it means the specific file wasn't there.
    mock_datasources.no_sql_profiles_client.find_one.reset_mock()
    mock_datasources.no_sql_profiles_client.find_one.return_value = {"_id": profile_id, "files": []}
    with pytest.raises(Exception) as excinfo:
        profile_repository.get_file_by_id(profile_id, file_id_to_find)
    assert f"File with ID {file_id_to_find} not found in profile {profile_id}" in str(excinfo.value)


def test_get_file_by_id_pymongo_error(profile_repository: Repository, mock_datasources: MagicMock):
    profile_id = "profile_get_file_err"
    file_id = "file_err_id_get"
    mock_datasources.no_sql_profiles_client.find_one.side_effect = PyMongoError("DB error")

    with pytest.raises(RuntimeError) as excinfo:
        profile_repository.get_file_by_id(profile_id, file_id)
    assert f"Error getting file {file_id} from profile {profile_id}" in str(excinfo.value)

# Final check for imports, already handled at the top
# from src.core.domain.model import Profile, ApiKey, File, datetime, uuid4, Optional, List # These are illustrative
# from pymongo.errors import PyMongoError, DuplicateKeyError # Illustrative
# import pytest # Illustrative
# from unittest.mock import MagicMock # Illustrative
