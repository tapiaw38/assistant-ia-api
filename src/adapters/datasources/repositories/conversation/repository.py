from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.domain.model import (
    Conversation,
    Message,
    Profile,
)
from src.adapters.datasources.datasources import Datasources
from pymongo.errors import PyMongoError


class RepositoryInterface(ABC):
    @abstractmethod
    def create(self, conversation: Conversation) -> str:
        pass

    @abstractmethod
    def find_by_id(self, id: str) -> Conversation:
        pass

    @abstractmethod
    def find_user_id(self, user_id: str) -> List[Conversation]:
        pass

    def update_profile_by_user_id(self, user_id: str, profile: Profile) -> str:
        pass

    @abstractmethod
    def update_messages_by_id(self, id: str, Messsages: List[Message]) -> None:
        pass

class Repository(RepositoryInterface):
    def __init__(self, datasources: Datasources):
        self.client = datasources.no_sql_conversations_client

    def create(self, conversation: Conversation) -> str:
        try:
            conversation_dict = conversation.dict(by_alias=True, exclude_none=True, exclude_unset=True)
            result = self.client.insert_one(conversation_dict)
            conversation_id = str(result.inserted_id)
            return conversation_id
        except PyMongoError as e:
            raise Exception(f"Error creating conversation: {e}")

    def find_by_id(self, id: str) -> Optional[Conversation]:
        try:
            document = self.client.find_one({"_id": id})
            if document:
                return Conversation(**document)
            return None
        except PyMongoError as e:
            raise Exception(f"Error finding conversation by ID: {e}")

    def find_user_id(self, user_id: str) -> List[Conversation]:
        try:
            documents = self.client.find({"profile.user_id": user_id}).sort("created_at", -1)
            return [
                Conversation(**{**doc, "id": str(doc["_id"])} ) for doc in documents
            ]
        except PyMongoError as e:
            raise Exception(f"Error finding conversations by user_id: {e}")

    def update_profile_by_user_id(self, user_id: str, profile: Profile) -> str:
        try:
            profile_data = profile.dict(by_alias=True, exclude_none=True, exclude_unset=True)

            result = self.client.update_many(
                {"profile.user_id": user_id},
                {"$set": {"profile": profile_data}}
            )

            if result.matched_count == 0:
                raise Exception(f"No conversation found with user_id: {user_id}")

            return user_id

        except PyMongoError as e:
            raise Exception(f"Error updating conversation: {e}")


    def update_messages_by_id(self, id: str, messages: List[Message]) -> None:
        try:
            messages_dict = [message.dict(by_alias=True) for message in messages]
            result = self.client.update_one(
                {"_id": id},
                {"$set": {"messages": messages_dict}}
            )

            if result.matched_count == 0:
                raise Exception(f"No conversation found with id: {id}")

        except PyMongoError as e:
            raise Exception(f"Error updating conversation: {e}")