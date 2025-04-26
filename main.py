from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from src.routes.routes_manager import RoutesManager
from src.adapters.integrations.integrations import create_integrations

app = FastAPI(
    title="Assistant IA API",
    description="API for the Assistant IA",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

routes_manager = RoutesManager(app)
routes_manager.include_routes()