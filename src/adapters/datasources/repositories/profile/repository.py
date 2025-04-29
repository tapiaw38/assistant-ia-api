from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.domain.model import (
    Profile,
    ApiKey,
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
    def update(self, profile: Profile) -> str:
        pass

    @abstractmethod
    def generate_api_key(self, api_keys: list[ApiKey]) -> None:
        pass


class Repository(RepositoryInterface):
    def __init__(self, datasources: Datasources):
        self.client = datasources.no_sql_profiles_client
        self.client.create_index("user_id", unique=True)

    def create(self, profile: Profile) -> str:
        try:
            profile_dict = profile.dict(by_alias=True)
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

    def update(self, id: str, profile: Profile) -> str:
        try:
            profile_dict = profile.dict(exclude_none=True, exclude_unset=True)
            result = self.client.update_one(
                {"_id": id},
                {"$set": profile_dict},
            )

            if result.matched_count == 0:
                raise Exception(f"No profile found with id: {profile.id}")

            return profile.id

        except PyMongoError as e:
            raise Exception(f"Error updating profile: {e}")

    def generate_api_key(self, id: str, api_keys: list[ApiKey]) -> None:
        try:
            api_keys_dict = [api_key.dict(by_alias=True) for api_key in api_keys]
            result = self.client.update_one(
                {"_id": id},
                {"$set": {"api_keys": api_keys_dict}}
            )

            if result.matched_count == 0:
                raise Exception(f"No profile found with id: {id}")

        except PyMongoError as e:
            raise Exception(f"Error updating profile: {e}")