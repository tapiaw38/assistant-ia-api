from fastapi import FastAPI
from src.adapters.controllers.conversation.controllers import router as conversation_router
from src.core.use_cases.use_cases import Usecases
from adapters.services.conversation.service import ConversationService
from src.adapters.services.services import (
    Services
)
class RoutesManager:
    def __init__(self, app: FastAPI, usecases: Usecases):
        self.app = app
        self.usecases = usecases

    def include_routes(self):
        conversation_service = ConversationService(self.usecases.conversation)

        services = Services.create_services(conversation=conversation_service)

        self.app.dependency_overrides[Services.get_instance] = lambda: services

        self.app.include_router(conversation_router)