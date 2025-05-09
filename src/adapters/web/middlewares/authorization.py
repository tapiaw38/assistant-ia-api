from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import jwt
from jwt import PyJWTError
from src.core.platform.config.service import get_config_service
import logging


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


async def validate_api_key(token: str) -> Optional[str]:
    secret_key = get_config_service().server_config.jwt_secret
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        if payload.get("type") != "api_key":
            return None
        print("API key valid")
        print(payload)
        return payload
    except jwt.ExpiredSignatureError:
        print("API key expired")
        return None
    except jwt.InvalidTokenError:
        print("Invalid API key")
        return None


class AuthorizationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXCLUDED_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        authorization: Optional[str] = request.headers.get("Authorization")

        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split("bearer ")[1].strip()
            request.state.user = await decode_token(token)
            return await call_next(request)

        api_key = request.headers.get("x-api-key")
        if api_key:
            request.state.user = await validate_api_key(api_key)
            return await call_next(request)

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token/api-key")