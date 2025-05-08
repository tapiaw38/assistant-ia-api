from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from src.routes.routes_manager import RoutesManager
from src.adapters.web.integrations.integrations import create_integrations
from src.core.platform.config.service import (
    init_config,
    get_config_service
)
from src.core.platform.nosql.client import MongoDBClient
from src.adapters.datasources.datasources import Datasources
from src.core.platform.appcontext.appcontext import new_factory
from src.core.use_cases.use_cases import create_usecases
from src.core.platform.nosql.migrations import execute_profile_migrations
from src.adapters.web.middlewares.authorization import authorization_middleware
from fastapi.middleware.base import BaseHTTPMiddleware


app = FastAPI(
    title="Assistant IA API",
    description="API for the Assistant IA",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://assistant-ia-fe.s3-website-sa-east-1.amazonaws.com",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

app.add_middleware(BaseHTTPMiddleware, dispatch=authorization_middleware)

init_config()
config_service = get_config_service()


migrations_client = MongoDBClient(
    config_service.nosql_config.migrations.database_uri,
    config_service.nosql_config.migrations.database,
    config_service.nosql_config.migrations.collection,
)
conversations_client = MongoDBClient(
    config_service.nosql_config.conversations.database_uri,
    config_service.nosql_config.conversations.database,
    config_service.nosql_config.conversations.collection,
)
profiles_client = MongoDBClient(
    config_service.nosql_config.profiles.database_uri,
    config_service.nosql_config.profiles.database,
    config_service.nosql_config.profiles.collection,
)

profiles_client.run_migrations(migrations_client.get_collection(), execute_profile_migrations())

datasources = Datasources.create_datasources(
    no_sql_hotel_client=conversations_client,
    no_sql_locations_client=profiles_client,
)
integrations = create_integrations(config_service)
context_factory = new_factory(datasources, integrations, config_service)
usecases = create_usecases(context_factory)

routes_manager = RoutesManager(app, usecases)
routes_manager.include_routes()