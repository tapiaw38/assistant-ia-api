from fastapi import Request, HTTPException, Response, status
from typing import Optional
import jwt
from jwt import PyJWTError
from src.core.platform.config.service import get_config_service


async def decode_token(token: str):
    try:
        payload = jwt.decode(
            token,
            get_config_service().server_config.jwt_secret,
            algorithms=[get_config_service().server_config.encryption_algorithm],
        )
        return payload
    except PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def authorization_middleware(request: Request, call_next):
    authorization: Optional[str] = request.headers.get("Authorization")
    if not authorization:
        return Response("Token is missing", status_code=status.HTTP_401_UNAUTHORIZED)

    if not authorization.startswith("bearer "):
        return Response("Invalid token format", status_code=status.HTTP_401_UNAUTHORIZED)

    token = authorization.split("bearer ")[1].strip()

    try:
        request.state.user = await decode_token(token)
    except PyJWTError:
        return Response("Invalid token", status_code=status.HTTP_401_UNAUTHORIZED)

    return await call_next(request)
