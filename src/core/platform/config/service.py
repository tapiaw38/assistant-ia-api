import os
from dotenv import load_dotenv
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ModeServer(str, Enum):
    RELEASE = "release"
    DEBUG = "debug"


@dataclass
class NoSQLCollectionConfig:
    database_uri: str
    database: str
    collection: str

@dataclass
class NoSQLConfig:
    migrations: Optional[NoSQLCollectionConfig] = None
    conversations: Optional[NoSQLCollectionConfig] = None
    profiles: Optional[NoSQLCollectionConfig] = None

@dataclass
class ServerConfig:
    mode: ModeServer
    port: str
    host: str
    jwt_secret: str
    encryption_algorithm: str

@dataclass
class OpenAIConfig:
    api_key: str
    base_url: str
    model: str
    role: str

@dataclass
class ConfigurationService:
    server_config: ServerConfig
    nosql_config: NoSQLConfig
    openai_config: OpenAIConfig

_config_service: Optional[ConfigurationService] = None

def init_config_service(config: ConfigurationService):
    global _config_service
    if _config_service is None:
        _config_service = config

def get_config_service() -> ConfigurationService:
    if _config_service is None:
        raise ValueError("ConfigurationService has not been initialized.")
    return _config_service

def get_env(key: str, fallback: str) -> str:
    return os.getenv(key, fallback)


def read_config() -> ConfigurationService:
    load_dotenv()

    db_user = get_env("DB_USER", "admin")
    db_password = get_env("DB_PASSWORD", "admin")
    db_host = get_env("DB_HOST", "localhost")
    db_port = get_env("DB_PORT", "27017")
    db_name = get_env("DB_NAME", "assistant-ia-db")

    database_uri = f"mongodb://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    server_config = ServerConfig(
        mode=ModeServer(get_env("APP_MODE", "release")),
        port=get_env("APP_PORT", "8000"),
        host=get_env("APP_HOST", "localhost"),
        jwt_secret=get_env("APP_JWT_SECRET", "secret"),
        encryption_algorithm=get_env("APP_ENCRYPTION_ALGORITHM", "HS256"),
    )

    nosql_config = NoSQLConfig(
        migrations=NoSQLCollectionConfig(
            database_uri=database_uri,
            database=get_env("DB_NAME", "assistant-ia-db"),
            collection=get_env("DB_COLLECTION_MIGRATIONS", "migrations"),
        ),
        conversations=NoSQLCollectionConfig(
            database_uri=database_uri,
            database=get_env("DB_NAME", "assistant-ia-db"),
            collection=get_env("DB_COLLECTION_CONVERSATIONS", "conversations"),
        ),
        profiles=NoSQLCollectionConfig(
            database_uri=database_uri,
            database=get_env("DB_NAME", "assistant-ia-db"),
            collection=get_env("DB_COLLECTION_PROFILES", "profiles"),
        ),
    )

    openai_config = OpenAIConfig(
        api_key=get_env("OPENAI_API_KEY", ""),
        base_url=get_env("OPENAI_BASE_URL", "https://api.deepseek.com"),
        model=get_env("OPENAI_MODEL", "deepseek-chat"),
        role=get_env("OPENAI_ROLE", "user"),
    )

    return ConfigurationService(
        server_config=server_config, 
        nosql_config=nosql_config,
        openai_config=openai_config,
    )


def init_config() -> None:
    config_service = read_config()
    init_config_service(config_service)