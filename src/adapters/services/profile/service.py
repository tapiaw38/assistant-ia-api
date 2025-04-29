from fastapi import HTTPException, status
from src.schemas.schemas import (
    ProfileInput,
)
from src.core.use_cases.use_cases import Profile


class ProfileService:
    def __init__(self, usecase: Profile):
        self.usecase = usecase

    async def create(self, profile: ProfileInput):
        try:
            created_profile = self.usecase.create_usecase.execute(profile)

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

    async def update(self, id: str, profile: ProfileInput):
        try:
            profile = self.usecase.update_usecase.execute(id, profile)
            return profile
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

