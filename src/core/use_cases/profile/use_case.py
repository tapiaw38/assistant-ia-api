from src.schemas.schemas import (
    ProfileInput, ProfileOutput, ApiKeyInput, ApiKeyListOutput,ApiKeyDeleteOutput
)
from src.core.domain.model import (
    Profile,
    ApiKey,
)
from datetime import datetime, timezone
from pymongo.errors import PyMongoError
from src.core.platform.appcontext.appcontext import Factory
from uuid import uuid4
import jwt
from datetime import datetime, timezone, timedelta

class CreateUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, profile: ProfileInput, user_id: str):
        try:
            generated_id = str(uuid4())
            new_profile = Profile(
                _id= generated_id,
                user_id=user_id,
                assistant_name=profile.assistant_name,
                business_name=profile.business_name,
                functions=profile.functions,
                business_context=profile.business_context,
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )

            profile_id = self.context_factory.repositories.profile.create(new_profile)
            created_profile = self.context_factory.repositories.profile.find_by_id(profile_id)

            if not created_profile:
                raise Exception("Error profile not found after creation")

            return ProfileOutput.from_output(created_profile)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing CreateUseCase: {e}")


class FindByUserIdUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, user_id: str):
        try:
            profile = self.context_factory.repositories.profile.find_by_user_id(user_id)
            if not profile:
                raise Exception(f"No profile found with user_id: {user_id}")
            return ProfileOutput.from_output(profile)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing FindByUserIdUseCase: {e}")


class FindAllUserIdsUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self):
        try:
            user_ids = self.context_factory.repositories.profile.find_all_user_ids()
            return user_ids
        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing FindAllUserIdsUseCase: {e}")


class UpdateUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, user_id: str, profile: ProfileInput):
        try:
            updated_profile = Profile(
                assistant_name=profile.assistant_name,
                business_name=profile.business_name,
                functions=profile.functions,
                business_context=profile.business_context,
                updated_at=datetime.now(timezone.utc),
            )

            search_profile = self.context_factory.repositories.profile.find_by_user_id(user_id)
            if not search_profile:
                raise Exception(f"No profile found with user_id: {user_id}")

            profile_updated_id = self.context_factory.repositories.profile.update(search_profile.id, updated_profile)
            if not profile_updated_id:
                raise Exception(f"No profile found with id: {search_profile.id}")

            profile_updated = self.context_factory.repositories.profile.find_by_id(profile_updated_id)
            if not profile_updated:
                raise Exception(f"No profile found with user_id: {user_id} after update")

            return ProfileOutput.from_output(profile_updated)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing UpdateUseCase: {e}")


class ChangeStatusUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, user_id: str, status: bool):
        try:
            updated_profile = Profile(
                updated_at=datetime.now(timezone.utc),
                is_active=status,
            )

            search_profile = self.context_factory.repositories.profile.find_by_user_id(user_id)
            if not search_profile:
                raise Exception(f"No profile found with user_id: {user_id}")

            profile_updated_id = self.context_factory.repositories.profile.update(search_profile.id, updated_profile)
            if not profile_updated_id:
                raise Exception(f"No profile found with id: {search_profile.id}")

            profile_updated = self.context_factory.repositories.profile.find_by_id(profile_updated_id)
            if not profile_updated:
                raise Exception(f"No profile found with user_id: {user_id} after update")

            return ProfileOutput.from_output(profile_updated)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing ChangeStatusUseCase: {e}")


class AddApiKeyUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def generate_api_key(self, user_id: str, expires_in_minutes: int = 60*24*365) -> str:
        secret_key = self.context_factory.config_service.server_config.jwt_secret
        payload = {
            "user_id": user_id,
            "type": "api_key",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes),
            "iat": datetime.now(timezone.utc)
        }

        token = jwt.encode(payload, secret_key, algorithm="HS256")
        return token

    async def execute(self, api_key: ApiKeyInput, user_id: str):
        try:
            generated_id = str(uuid4())
            new_api_key = ApiKey(
                _id=generated_id,
                user_id=user_id,
                value=self.generate_api_key(user_id),
                description=api_key.description,
                limit=api_key.limit or 1000,
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )

            profile = self.context_factory.repositories.profile.find_by_user_id(user_id)
            if not profile:
                raise Exception(f"No profile found with user_id: {user_id}")

            if not hasattr(profile, "api_keys") or profile.api_keys is None:
                profile.api_keys = []

            if any(api_key.id == new_api_key.id for api_key in profile.api_keys):
                raise Exception("API key with the same ID already exists in the profile")

            profile.api_keys.append(new_api_key)

            self.context_factory.repositories.profile.update_api_keys(user_id, profile.api_keys)

            return ApiKeyListOutput.from_output(profile.api_keys)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing AddApiKeyUseCase: {e}")


class DeleteApiKeyUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    async def execute(self, user_id: str, api_key_id: str):
        try:
            self.context_factory.repositories.profile.delete_api_key(user_id, api_key_id)

            return ApiKeyDeleteOutput.from_output(api_key_id)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing DeleteApiKeyUseCase: {e}")