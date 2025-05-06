from src.schemas.schemas import (
    ProfileInput, ProfileOutput
)
from src.core.domain.model import (
    Profile,
)
from datetime import datetime, timezone
from pymongo.errors import PyMongoError
from src.core.platform.appcontext.appcontext import Factory
from uuid import uuid4


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