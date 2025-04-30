from fastapi import APIRouter, status, Depends
from src.schemas.schemas import (
    ProfileInput,
    ProfileStatusInput,
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
    mock_user_id = "mock_user_id" # TODO: replace with user_id in request
    profile = await service.profile.create(profile, mock_user_id)
    return profile


@router.get("/", status_code=status.HTTP_200_OK)
async def find_profile_by_user_id(
    service: Services = Depends(get_instance)
):
    mock_user_id = "mock_user_id" # TODO: replace with user_id in request
    profile = await service.profile.find_by_user_id(mock_user_id)
    return profile

@router.patch("/", status_code=status.HTTP_200_OK)
async def update_profile(
    profile: ProfileInput,
    service: Services = Depends(get_instance)
):
    mock_user_id = "mock_user_id" # TODO: replace with user_id in request
    profile = await service.profile.update(mock_user_id, profile)
    return profile


@router.post("/status", status_code=status.HTTP_200_OK)
async def change_status(
    status_input: ProfileStatusInput,
    service: Services = Depends(get_instance)
):
    mock_user_id = "mock_user_id"  # TODO: replace with user_id in request
    profile = await service.profile.change_status(mock_user_id, status_input.is_active)
    return profile