from fastapi import APIRouter, status, Depends, Request
from src.schemas.schemas import (
    ProfileInput,
    ProfileStatusInput,
    ApiKeyInput,
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
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    print(user_id)
    profile = await service.profile.create(profile, user_id)
    return profile


@router.get("/", status_code=status.HTTP_200_OK)
async def find_profile_by_user_id(
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    profile = await service.profile.find_by_user_id(user_id)
    return profile

@router.patch("/", status_code=status.HTTP_200_OK)
async def update_profile(
    profile: ProfileInput,
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    profile = await service.profile.update(user_id, profile)
    return profile


@router.post("/status", status_code=status.HTTP_200_OK)
async def change_status(
    status_input: ProfileStatusInput,
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    profile = await service.profile.change_status(user_id, status_input.is_active)
    return profile


@router.post("/api-key", status_code=status.HTTP_201_CREATED)
async def add_api_key(
    api_key: ApiKeyInput,
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    api_keys = await service.profile.add_api_key(user_id, api_key)
    return api_keys


@router.delete("/api-key/{api_key_id}", status_code=status.HTTP_200_OK)
async def delete_api_key(
    api_key_id: str,
    request: Request,
    service: Services = Depends(get_instance)
):
    user_id = request.state.user.get("user_id")
    api_key_id = await service.profile.delete_api_key(user_id, api_key_id)
    return api_key_id