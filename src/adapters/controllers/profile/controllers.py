from fastapi import APIRouter, status, Depends
from src.schemas.schemas import (
    ProfileInput,
)
from src.adapters.services.services import Services


router = APIRouter(
    prefix="/profile",
    tags=["profile"],
)

def get_instance() -> Services:
    return Services.get_instance()

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_profile(
    profile: ProfileInput,
    service: Services = Depends(get_instance)
):
    profile = await service.profile.create(profile)
    return profile


@router.get("/user/{user_id}", status_code=status.HTTP_200_OK)
async def find_profile_by_user_id(
    user_id: str,
    service: Services = Depends(get_instance)
):
    profile = await service.profile.find_by_user_id(user_id)
    return profile

@router.post("/{id}", status_code=status.HTTP_201_CREATED)
async def update_profile(
    id: str,
    profile: ProfileInput,
    service: Services = Depends(get_instance)
):
    profile = await service.profile.update(id, profile)
    return profile