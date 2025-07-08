from src.adapters.web.integrations.openapi.openapi import OpenAIIntegration
from src.adapters.web.integrations.whatsapp.whatsapp import WhatsAppIntegration
from src.core.platform.config.service import ConfigurationService
from src.core.domain.model import Integration as IntegrationModel
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Integrations:
    _instance = None

    def __init__(self, config_service: ConfigurationService):
        if Integrations._instance is not None:
            raise Exception("This class is a singleton! Use `Integrations.get_instance()` to access it.")
        self.openai = OpenAIIntegration(config_service)
        self.whatsapp_instances: Dict[str, WhatsAppIntegration] = {}
        Integrations._instance = self

    @classmethod
    def get_instance(cls) -> "Integrations":
        if cls._instance is None:
            raise Exception("Integrations has not been initialized. Call `create_integrations` first.")
        return cls._instance

    def get_whatsapp_integration(self, integration: IntegrationModel) -> Optional[WhatsAppIntegration]:
        """
        Obtiene una instancia de WhatsAppIntegration basada en la configuración
        """
        try:
            if integration.type != "whatsapp":
                logger.warning(f"Integration type {integration.type} is not WhatsApp")
                return None
            
            config = integration.config
            if not config.get("access_token") or not config.get("phone_number_id"):
                logger.error("WhatsApp integration missing required configuration")
                return None
            
            # Usar un cache de instancias para evitar recrear la integración
            cache_key = f"{integration.id}_{config['phone_number_id']}"
            
            if cache_key not in self.whatsapp_instances:
                self.whatsapp_instances[cache_key] = WhatsAppIntegration(
                    access_token=config["access_token"],
                    phone_number_id=config["phone_number_id"]
                )
                logger.info(f"Created new WhatsApp integration instance for {cache_key}")
            
            return self.whatsapp_instances[cache_key]
        
        except Exception as e:
            logger.error(f"Error creating WhatsApp integration: {e}")
            return None

    def remove_whatsapp_integration(self, integration_id: str):
        """
        Remueve una instancia de WhatsApp del cache
        """
        keys_to_remove = [key for key in self.whatsapp_instances.keys() if key.startswith(f"{integration_id}_")]
        for key in keys_to_remove:
            del self.whatsapp_instances[key]
            logger.info(f"Removed WhatsApp integration instance {key}")

    def validate_whatsapp_config(self, config: Dict[str, Any]) -> bool:
        """
        Valida la configuración de WhatsApp
        """
        required_fields = ["access_token", "phone_number_id"]
        optional_fields = ["webhook_verify_token", "app_secret"]
        
        # Verificar campos requeridos
        for field in required_fields:
            if not config.get(field):
                logger.error(f"WhatsApp configuration missing required field: {field}")
                return False
        
        # Verificar que los campos tengan el formato correcto
        if not isinstance(config["access_token"], str) or len(config["access_token"]) < 10:
            logger.error("WhatsApp access_token is invalid")
            return False
        
        if not isinstance(config["phone_number_id"], str) or not config["phone_number_id"].isdigit():
            logger.error("WhatsApp phone_number_id is invalid")
            return False
        
        logger.info("WhatsApp configuration is valid")
        return True


def create_integrations(cfg: ConfigurationService) -> Integrations:
    if Integrations._instance is None:
        return Integrations(cfg)
    return Integrations.get_instance()