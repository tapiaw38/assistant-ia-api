from dotenv import load_dotenv
import os

from src.core.platform.nosql.migrations import Migration
from src.core.platform.nosql.client import MongoDBClient
from src.core.platform.config.service import (
    ConfigurationService,
    ServerConfig,
    ModeServer,
    NoSQLConfig,
    OpenAIConfig,
    init_config_service,
)


def init_config() -> None:
    config_service = read_config()
    init_config_service(config_service)

def read_config() -> ConfigurationService:
    load_dotenv()

    config_service = ConfigurationService(
        server_config=ServerConfig(
            mode=ModeServer(os.getenv("APP_MODE", "release")),
            port=os.getenv("APP_PORT", "8000"),
            host=os.getenv("APP_HOST", "localhost"),
            jwt_secret=os.getenv("APP_JWT_SECRET", "secret"),
        ),
        nosql_config=NoSQLConfig(
            migrations=NoSQLConfig(
                database_uri=os.getenv("DATABASE_URI", "mongodb://localhost:27017"),
                database=os.getenv("DB_NAME", "assistant-ia-db"),
                collection=os.getenv("DB_COLLECTION_MIGRATIONS", "migrations"),
            ),
            conversations=NoSQLConfig(
                database_uri=os.getenv("DATABASE_URI", "mongodb://localhost:27017"),
                database=os.getenv("DB_NAME", "assistant-ia-db"),
                collection=os.getenv("DB_COLLECTION_CONVERSATIONS", "conversations"),
            ),
            profiles=NoSQLConfig(
                database_uri=os.getenv("DATABASE_URI", "mongodb://localhost:27017"),
                database=os.getenv("DB_NAME", "assistant-ia-db"),
                collection=os.getenv("DB_COLLECTION_PROFILES", "profiles"),
            ),
        ),
        openai_config=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", "sk-dce04e45febe486bbeb49b452d608e07"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            role=os.getenv("OPENAI_ROLE", "user"),
        ),
    )

    return config_service