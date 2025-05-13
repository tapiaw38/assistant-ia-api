from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import jwt
from jwt import PyJWTError
from src.core.platform.config.service import get_config_service
from src.adapters.services.services import Services


EXCLUDED_PATHS = {"/docs", "/docs/", "/openapi.json", "/redoc", "/redoc/"}

async def decode_token(token: str):
    try:
        config = get_config_service().server_config
        payload = jwt.decode(
            token,
            config.jwt_secret,
            algorithms=[config.encryption_algorithm],
        )
        return payload
    except PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def validate_api_key(token: str, services: Services) -> Optional[str]:
    secret_key = get_config_service().server_config.jwt_secret
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        if payload.get("type") != "api_key":
            return None

        user_id = payload.get("user_id")
        api_key = await services.profile.find_by_api_key_value(user_id, token)

        if not api_key or not api_key.is_active:
            return None

        profile = await services.profile.find_by_user_id(user_id)
        if not profile or not profile.data.is_active:
            return None

        if profile.data.iteration_limit is not None and profile.data.iteration_limit <= 0:
            return None

        updated_iteration_limit = profile.data.iteration_limit - 1
        await services.profile.update_iteration_limit(user_id, updated_iteration_limit)

        return payload

    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


class AuthorizationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, services: Services):
        super().__init__(app)
        self.get_instance = services.get_instance()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXCLUDED_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        authorization: Optional[str] = request.headers.get("Authorization")

        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split("bearer ")[1].strip()
            user_id = await decode_token(token)
            if not user_id:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token/api-key")
            request.state.user = user_id
            return await call_next(request)

        api_key = request.headers.get("x-api-key")
        if api_key:
            user_id = await validate_api_key(api_key, self.get_instance)
            if not user_id:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token/api-key")
            request.state.user = user_id
            return await call_next(request)

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token/api-key")