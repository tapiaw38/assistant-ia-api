from fastapi import HTTPException, status
from src.schemas.schemas import (
    ProfileInput,
    ApiKeyInput,
)
from src.core.use_cases.use_cases import Profile


class ProfileService:
    def __init__(self, usecase: Profile):
        self.usecase = usecase

    async def create(self, profile: ProfileInput, user_id: str):
        try:
            created_profile = self.usecase.create_usecase.execute(profile, user_id)

            if created_profile is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile not created")

            return created_profile

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def find_by_user_id(self, user_id: str):
        try:
            profile = self.usecase.find_by_user_id_usecase.execute(user_id)
            return profile
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def find_all_user_ids(self):
        try:
            user_ids = self.usecase.find_all_user_ids_usecase.execute()
            return user_ids
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def update(self, user_id: str, profile: ProfileInput):
        try:
            profile = self.usecase.update_usecase.execute(user_id, profile)
            return profile
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def change_status(self, user_id: str, status: bool):
        try:
            profile = self.usecase.change_status_usecase.execute(user_id, status)
            return profile
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def add_api_key(self, user_id: str, api_key: ApiKeyInput):
        try:
            api_keys = await self.usecase.add_api_key_usecase.execute(api_key, user_id)
            return api_keys
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def delete_api_key(self, user_id: str, api_key_id: str):
        try:
            api_key_id = await self.usecase.delete_api_key_usecase.execute(user_id, api_key_id)
            return api_key_id
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))