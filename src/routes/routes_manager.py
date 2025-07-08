from fastapi import FastAPI
from src.adapters.web.controllers.conversation.controllers import router as conversation_router
from src.adapters.web.controllers.profile.controllers import router as profile_router
from src.adapters.web.controllers.whatsapp.controllers import router as whatsapp_router
from src.adapters.web.controllers.whatsapp.webhooks import router as whatsapp_webhooks_router
from src.core.use_cases.use_cases import Usecases
from src.adapters.services.conversation.service import ConversationService
from src.adapters.services.profile.service import ProfileService
from src.adapters.services.services import (
    Services
)
from src.adapters.web.middlewares.authorization import AuthorizationMiddleware


class RoutesManager:
    def __init__(self, app: FastAPI, usecases: Usecases):
        self.app = app
        self.usecases = usecases

    def include_routes(self):
        conversation_service = ConversationService(self.usecases.conversation)
        profile_service = ProfileService(self.usecases.profile)

        services = Services.create_services(
            conversation=conversation_service,
            profile=profile_service,
        )

        self.app.dependency_overrides[Services.get_instance] = lambda: services

        self.app.add_middleware(AuthorizationMiddleware, services=Services.get_instance())
        self.app.include_router(profile_router)
        self.app.include_router(conversation_router)
        self.app.include_router(whatsapp_router)
        self.app.include_router(whatsapp_webhooks_router)