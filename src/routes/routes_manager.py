from fastapi import FastAPI
from src.adapters.controllers.controllers import router as conversation_router

class RoutesManager:
    def __init__(self, app: FastAPI):
        self.app = app

    def include_routes(self):
        self.app.include_router(conversation_router)