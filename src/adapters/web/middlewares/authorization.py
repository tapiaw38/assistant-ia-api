from fastapi import Request, HTTPException, status
from typing import Optional
import jwt
from jwt import PyJWTError
from src.core.platform.config.service import get_config_service


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

EXCLUDED_PATHS = {"/docs", "/docs/", "/openapi.json", "/redoc", "/redoc/"}

async def authorization_middleware(request: Request, call_next):
    if request.url.path in EXCLUDED_PATHS:
        return await call_next(request)

    if request.method == "OPTIONS":
        return await call_next(request)

    authorization: Optional[str] = request.headers.get("Authorization")
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is missing")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")

    token = authorization.split("bearer ")[1].strip()

    request.state.user = await decode_token(token)

    return await call_next(request)
