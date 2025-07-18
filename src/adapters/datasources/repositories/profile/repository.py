from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.domain.model import (
    Profile,
    ApiKey,
    File,
    Integration,
)
from src.adapters.datasources.datasources import Datasources
from pymongo.errors import (
    PyMongoError,
    DuplicateKeyError,
)


class RepositoryInterface(ABC):
    @abstractmethod
    def create(self, profile: Profile) -> str:
        pass

    @abstractmethod
    def find_by_id(self, id: str) -> Profile:
        pass

    @abstractmethod
    def find_by_user_id(self, user_id: str) -> Profile:
        pass

    @abstractmethod
    def find_all_user_ids(self) -> List[str]:
        pass

    @abstractmethod
    def update(self, user_id: str, profile: Profile) -> str:
        pass

    @abstractmethod
    def update_api_keys(self, user_id: str, api_keys: List[ApiKey]) -> Optional[List[ApiKey]]:
        pass

    @abstractmethod
    def deactivate_api_key(self, user_id: str, api_key_id: str) -> Optional[str]:
        pass

    @abstractmethod
    def find_by_api_key_value(self, user_id: str, api_key_value: str) -> ApiKey:
        pass

    @abstractmethod
    def update_iteration_limit(self, user_id: str, iteration_limit: int) -> str:
        pass

    @abstractmethod
    def update_files(self, user_id: str, files: List[File]) -> None:
        pass

    @abstractmethod
    def delete_file_by_id(self, user_id: str, file_id: str) -> None:
        pass

    @abstractmethod
    def get_file_by_id(self, user_id: str, file_id: str) -> Optional[File]:
        pass

    @abstractmethod
    def update_integrations(self, user_id: str, integrations: List[Integration]) -> None:
        pass


class Repository(RepositoryInterface):
    def __init__(self, datasources: Datasources):
        self.client = datasources.no_sql_profiles_client
        self.client.create_index("user_id", unique=True)

    def create(self, profile: Profile) -> str:
        try:
            profile_dict = profile.dict(by_alias=True, exclude_unset=True, exclude_none=True)
            result = self.client.insert_one(profile_dict)
            profile_id = str(result.inserted_id)
            return profile_id

        except DuplicateKeyError as e:
            raise Exception(f"Error creating profile: User ID already exists")

        except PyMongoError as e:
            raise Exception(f"Error creating profile: {e}")

    def find_by_id(self, id: str) -> Optional[Profile]:
        try:
            document = self.client.find_one({"_id": id})
            if not document:
                raise Exception(f"No profile found with id: {id}")

            return Profile(**document)

        except PyMongoError as e:
            raise Exception(f"Error finding profile by ID: {e}")

    def find_by_user_id(self, user_id: str) -> Profile:
        try:
            document = self.client.find_one({"user_id": user_id})
            if not document:
                raise Exception(f"No profile found with user_id: {user_id}")

            return Profile(**document)

        except PyMongoError as e:
            raise Exception(f"Error finding profile by user_id: {e}")

    def find_all_user_ids(self) -> List[str]:
        try:
            documents = self.client.find({}, {"_id": 0, "user_id": 1})
            return [doc.get("user_id") for doc in documents if doc.get("user_id")]
        except PyMongoError as e:
            raise RuntimeError(f"Error retrieving user_ids from DB: {e}")

    def update(self, id: str, profile: Profile) -> str:
        try:
            profile_dict = profile.dict(
                exclude_none=True,
                by_alias=True
            )
            result = self.client.update_one(
                {"_id": id},
                {"$set": profile_dict},
            )

            if result.matched_count == 0:
                raise Exception(f"No profile found with id: {id}")

            return id

        except PyMongoError as e:
            raise Exception(f"Error updating profile: {e}")

    def update_api_keys(self, user_id: str, api_keys: List[ApiKey]) -> None:
        try:
            api_keys_dict = [api_key.dict(by_alias=True) for api_key in api_keys]
            result = self.client.update_one(
                {"user_id": user_id},
                {"$set": {"api_keys": api_keys_dict}}
            )

            if result.matched_count == 0:
                raise Exception(f"No profile found with user_id: {user_id}")

        except PyMongoError as e:
            raise Exception(f"Error updating profile api_keys: {e}")

    def deactivate_api_key(self, user_id: str, api_key_id: str) -> Optional[str]:
        try:
            result = self.client.update_one(
                {
                    "user_id": user_id,
                    "api_keys._id": api_key_id
                },
                {
                    "$set": {
                        "api_keys.$.is_active": False
                    }
                }
            )

            if result.modified_count == 0:
                raise Exception("API key not found or already inactive.")

            return api_key_id

        except PyMongoError as e:
            raise Exception(f"Error deactivating API key: {e}")

    def find_by_api_key_value(self, user_id: str, api_key_value: str) -> ApiKey:
        try:
            document = self.client.find_one(
                {
                    "user_id": user_id,
                    "api_keys.value": api_key_value
                },
                {
                    "api_keys": {
                        "$elemMatch": {"value": api_key_value}
                    }
                }
            )
            if not document or "api_keys" not in document or not document["api_keys"]:
                raise Exception(f"No API key found for user_id: {user_id}")

            return ApiKey(**document["api_keys"][0])

        except PyMongoError as e:
            raise Exception(f"Error finding API key: {e}")

    def update_iteration_limit(self, user_id: str, iteration_limit: int) -> str:
        try:
            result = self.client.update_one(
                {"user_id": user_id},
                {"$set": {"iteration_limit": iteration_limit}}
            )

            if result.matched_count == 0:
                raise Exception(f"No profile found with user_id: {user_id}")

            return user_id

        except PyMongoError as e:
            raise Exception(f"Error updating profile: {e}")

    def update_files(self, user_id: str, files: List[File]) -> None:
        try:
            files_dict = [file.dict(by_alias=True) for file in files]
            result = self.client.update_one(
                {"user_id": user_id},
                {"$set": {"files": files_dict}}
            )

            if result.matched_count == 0:
                raise Exception(f"No profile found with user_id: {user_id}")

        except PyMongoError as e:
            raise Exception(f"Error updating profile files: {e}")

    def delete_file_by_id(self, user_id: str, file_id: str) -> None:
        try:
            result = self.client.update_one(
                {"user_id": user_id},
                {"$pull": {"files": {"_id": file_id}}}
            )

            if result.matched_count == 0:
                raise Exception(f"No profile found with user_id: {user_id}")

        except PyMongoError as e:
            raise Exception(f"Error deleting file by ID: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")

    def get_file_by_id(self, user_id: str, file_id: str) -> Optional[File]:
        try:
            document = self.client.find_one(
                {
                    "user_id": user_id,
                    "files._id": file_id
                },
                {
                    "files": {
                        "$elemMatch": {"_id": file_id}
                    }
                }
            )
            if not document or "files" not in document or not document["files"]:
                raise Exception(f"No file found with id: {file_id}")

            return File(**document["files"][0])

        except PyMongoError as e:
            raise Exception(f"Error finding file by ID: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")

    def update_integrations(self, user_id: str, integrations: List[Integration]) -> None:
        try:
            integrations_dict = [integration.dict(by_alias=True) for integration in integrations]
            result = self.client.update_one(
                {"user_id": user_id},
                {"$set": {"integrations": integrations_dict}}
            )

            if result.matched_count == 0:
                raise Exception(f"No profile found with user_id: {user_id}")

        except PyMongoError as e:
            raise Exception(f"Error updating profile integrations: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
