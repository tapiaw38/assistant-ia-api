from src.adapters.web.integrations.openapi.openapi import OpenAIIntegration
from src.adapters.web.integrations.mercadolibre.mercadolibre import MercadoLibreIntegration
from src.core.platform.config.service import ConfigurationService


class Integrations:
    _instance = None

    def __init__(self, config_service: ConfigurationService):
        if Integrations._instance is not None:
            raise Exception("This class is a singleton! Use `Integrations.get_instance()` to access it.")
        self.openai = OpenAIIntegration(config_service)
        self.mercadolibre = None  # Will be initialized when needed with user credentials
        Integrations._instance = self

    def initialize_mercadolibre(self, access_token: str, user_id: str):
        """Initialize MercadoLibre integration with user credentials"""
        self.mercadolibre = MercadoLibreIntegration(access_token, user_id)
        return self.mercadolibre

    @classmethod
    def get_instance(cls) -> "Integrations":
        if cls._instance is None:
            raise Exception("Integrations has not been initialized. Call `create_integrations` first.")
        return cls._instance


def create_integrations(cfg: ConfigurationService) -> Integrations:
    if Integrations._instance is None:
        return Integrations(cfg)
    return Integrations.get_instance()