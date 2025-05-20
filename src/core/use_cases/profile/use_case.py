from src.schemas.schemas import (
    ProfileInput, 
    ProfileOutput, 
    ApiKeyInput, 
    ApiKeyListOutput,
    ApiKeyDeleteOutput,
    FileListOutput,
    FileInput,
    FileDeleteOutput,
)
from src.core.domain.model import (
    Profile,
    ApiKey,
    File,
)
from datetime import datetime, timezone
from pymongo.errors import PyMongoError
from src.core.platform.appcontext.appcontext import Factory
from uuid import uuid4
import jwt
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional
from asyncio import to_thread, gather

class CreateUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    def execute(self, profile: ProfileInput, user_id: str):
        try:
            generated_id = str(uuid4())
            iteration_limit = 100
            new_profile = Profile(
                _id= generated_id,
                user_id=user_id,
                assistant_name=profile.assistant_name,
                business_name=profile.business_name,
                functions=profile.functions,
                business_context=profile.business_context,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                iteration_limit=iteration_limit,
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
            default_limit = 1000
            generated_id = str(uuid4())
            new_api_key = ApiKey(
                _id=generated_id,
                user_id=user_id,
                value=self.generate_api_key(user_id),
                description=api_key.description,
                limit=default_limit,
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
            self.context_factory.repositories.profile.deactivate_api_key(user_id, api_key_id)

            return ApiKeyDeleteOutput.from_output(api_key_id)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing DeleteApiKeyUseCase: {e}")

class FindByApiKeyValueUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    async def execute(self, user_id: str, api_key_value: str):
        try:
            api_key = self.context_factory.repositories.profile.find_by_api_key_value(user_id, api_key_value)
            if not api_key:
                raise Exception(f"No API key found for user_id: {user_id}")
            return api_key

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing FindByApiKeyValueUseCase: {e}")


class UpdateIterationLimitUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    async def execute(self, user_id: str, iteration_limit: int):
        try:
            profile_user_id = self.context_factory.repositories.profile.update_iteration_limit(user_id, iteration_limit)
            if not profile_user_id:
                raise Exception(f"No profile found with id: {profile_user_id}")

            profile_updated = self.context_factory.repositories.profile.find_by_user_id(profile_user_id)
            if not profile_updated:
                raise Exception(f"No profile found with user_id: {user_id} after update")

            return ProfileOutput.from_output(profile_updated)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing UpdateIterationLimitUseCase: {e}")


class AddFilesUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory

    async def execute(self, user_id: str, input_files: List[FileInput]):
        app = self.context_factory()

        results: List[File] = []
        errors: List[Exception] = []

        async def process_file(file: FileInput, user_file: str) -> Tuple[Optional[File], Optional[Exception]]:
            try:
                file_id = str(uuid4())

                file_name = await to_thread(
                    app.store_service.put_object, file.file, user_file, file.filename, file.filesize, file_id
                )
                file_url = await to_thread(app.store_service.generate_url, file_name)

                return File(
                    _id=file_id,
                    name=file_name,
                    url=file_url,
                    created_at=datetime.now(timezone.utc),
                ), None
            except Exception as e:
                return None, e

        tasks = [process_file(f, user_id) for f in input_files]
        results_and_errors = await gather(*tasks)

        for result, error in results_and_errors:
            if error:
                errors.append(error)
            elif result:
                results.append(result)

        if errors:
            raise Exception(f"Error processing files: {errors}")

        try:
            profile = app.repositories.profile.find_by_user_id(user_id)
            if not profile:
                raise Exception(f"No profile found with user_id: {user_id}")

            if not hasattr(profile, "files") or profile.files is None:
                profile.files = []

            profile.files.extend(results)

            app.repositories.profile.update_files(user_id, profile.files)

            updated_profile = app.repositories.profile.find_by_user_id(user_id)
            if not updated_profile:
                raise Exception(f"No profile found with user_id: {user_id} after update")

            return FileListOutput.from_output(updated_profile.files)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")
        except Exception as e:
            raise Exception(f"Error executing AddFilesUseCase: {e}")


class DeleteFileByIdUseCase:
    def __init__(self, context_factory: Factory):
        self.context_factory = context_factory()

    async def execute(self, user_id: str, file_id: str):
        try:
            file = self.context_factory.repositories.profile.get_file_by_id(user_id, file_id)
            if not file:
                raise Exception(f"No file found with id: {file_id}")

            self.context_factory.repositories.profile.delete_file_by_id(user_id, file_id)
            self.context_factory.store_service.delete_object(file.name)

            return FileDeleteOutput.from_output(file_id)

        except PyMongoError as e:
            raise Exception(f"Error interacting with database: {e}")

        except Exception as e:
            raise Exception(f"Error executing DeleteApiKeyUseCase: {e}")