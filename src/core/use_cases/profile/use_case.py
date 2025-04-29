from src.schemas.schemas import (
    ProfileInput,
)
from src.core.domain.model import (
    Profile,
)
from datetime import datetime, timezone
from pymongo.errors import PyMongoError
from src.core.platform.appcontext.appcontext import Factory


class CreateUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, profile: ProfileInput):
        try:
            new_profile = Profile(
                user_id=profile.user_id,
                assistant_name=profile.assistant_name,
                business_name=profile.business_name,
                prompt=profile.prompt,
                prompt_context=profile.prompt_context,
                created_at=datetime.now(timezone.utc),
            )

            profile_id = self.context_factory.repositories.profile.create(new_profile)

            created_profile =  self.context_factory.repositories.profile.find_by_id(profile_id)

            if not created_profile:
                raise Exception("Error profile not found after creation")

            return created_profile

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
            return profile
        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing FindByUserIdUseCase: {e}")


class UpdateUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, id: str, profile: ProfileInput):
        try:
            updated = Profile(
                assistant_name=profile.assistant_name,
                business_name=profile.business_name,
                prompt=profile.prompt,
                prompt_context=profile.prompt_context,
                updated_at=datetime.now(timezone.utc),
            )

            profile_updated_id = self.context_factory.repositories.profile.update(id, updated)
            if not profile_updated_id:
                raise Exception(f"No profile found with id: {id}")

            profile_updated = self.context_factory.repositories.profile.find_by_id(id)
            if not profile_updated:
                raise Exception(f"No profile found with id: {id} after update")

            return profile_updated

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")