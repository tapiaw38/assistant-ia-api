from typing import Dict, Any, List, Optional, Tuple
from src.adapters.web.integrations.whatsapp.whatsapp import WhatsAppIntegration
from src.core.domain.model import Profile, Integration, Message, Conversation
from src.schemas.schemas import WhatsAppMessageInput, WhatsAppTemplateMessageInput
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class WhatsAppService:
    """
    Servicio para manejar la lógica de negocio de WhatsApp
    """
    
    def __init__(self):
        self.active_integrations: Dict[str, WhatsAppIntegration] = {}
    
    def get_whatsapp_integration(self, profile: Profile) -> Optional[WhatsAppIntegration]:
        """
        Obtiene la integración de WhatsApp de un perfil
        """
        if not profile.integrations:
            return None
        
        whatsapp_integration = None
        for integration in profile.integrations:
            if integration.type == "whatsapp":
                whatsapp_integration = integration
                break
        
        if not whatsapp_integration:
            return None
        
        # Validar configuración
        config = whatsapp_integration.config
        if not config.get("access_token") or not config.get("phone_number_id"):
            logger.error(f"Invalid WhatsApp configuration for profile {profile.id}")
            return None
        
        # Crear o reutilizar instancia
        cache_key = f"{profile.id}_{integration.id}"
        if cache_key not in self.active_integrations:
            self.active_integrations[cache_key] = WhatsAppIntegration(
                access_token=config["access_token"],
                phone_number_id=config["phone_number_id"]
            )
        
        return self.active_integrations[cache_key]
    
    def validate_whatsapp_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Valida la configuración de WhatsApp
        """
        required_fields = ["access_token", "phone_number_id"]
        
        for field in required_fields:
            if not config.get(field):
                return False, f"Missing required field: {field}"
        
        # Validar formato del access_token
        if not isinstance(config["access_token"], str) or len(config["access_token"]) < 10:
            return False, "Invalid access_token format"
        
        # Validar formato del phone_number_id
        if not isinstance(config["phone_number_id"], str) or not config["phone_number_id"].isdigit():
            return False, "Invalid phone_number_id format"
        
        return True, "Configuration is valid"
    
    async def send_message(self, profile: Profile, message_data: WhatsAppMessageInput) -> Dict[str, Any]:
        """
        Envía un mensaje de WhatsApp
        """
        whatsapp_client = self.get_whatsapp_integration(profile)
        if not whatsapp_client:
            raise ValueError("WhatsApp integration not found or invalid")
        
        result = whatsapp_client.send_message(
            recipient=message_data.recipient,
            message=message_data.message,
            message_type=message_data.message_type
        )
        
        return result
    
    async def send_template_message(self, profile: Profile, template_data: WhatsAppTemplateMessageInput) -> Dict[str, Any]:
        """
        Envía un mensaje de plantilla de WhatsApp
        """
        whatsapp_client = self.get_whatsapp_integration(profile)
        if not whatsapp_client:
            raise ValueError("WhatsApp integration not found or invalid")
        
        result = whatsapp_client.send_template_message(
            recipient=template_data.recipient,
            template_name=template_data.template_name,
            language_code=template_data.language_code,
            parameters=template_data.parameters
        )
        
        return result
    
    async def process_incoming_message(self, webhook_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Procesa mensajes entrantes del webhook
        """
        try:
            # Crear instancia temporal para procesar el mensaje
            temp_whatsapp = WhatsAppIntegration(
                access_token="temp_token",
                phone_number_id="temp_phone_id"
            )
            
            messages = temp_whatsapp.receive_message(webhook_payload)
            processed_messages = []
            
            for message in messages:
                if message.get("type") == "status_update":
                    # Manejar actualizaciones de estado
                    logger.info(f"Status update: {message}")
                    continue
                
                # Procesar mensaje normal
                processed_message = await self._process_single_message(message)
                if processed_message:
                    processed_messages.append(processed_message)
            
            return processed_messages
            
        except Exception as e:
            logger.error(f"Error processing incoming message: {e}")
            raise
    
    async def _process_single_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Procesa un mensaje individual
        """
        try:
            from_number = message.get("from")
            message_text = message.get("text")
            message_type = message.get("type")
            message_id = message.get("id")
            timestamp = message.get("timestamp")
            
            if not from_number or not message_id:
                logger.warning(f"Invalid message format: {message}")
                return None
            
            # Aquí puedes implementar:
            # 1. Buscar el perfil del usuario basado en el número de teléfono
            # 2. Crear o continuar una conversación
            # 3. Generar respuesta usando IA
            # 4. Enviar respuesta automática
            
            processed_message = {
                "id": message_id,
                "from": from_number,
                "text": message_text,
                "type": message_type,
                "timestamp": timestamp,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "status": "processed"
            }
            
            logger.info(f"Processed message from {from_number}: {message_text}")
            return processed_message
            
        except Exception as e:
            logger.error(f"Error processing single message: {e}")
            return None
    
    async def create_conversation_from_whatsapp(self, profile: Profile, whatsapp_message: Dict[str, Any]) -> Optional[Conversation]:
        """
        Crea una conversación a partir de un mensaje de WhatsApp
        """
        try:
            from_number = whatsapp_message.get("from")
            message_text = whatsapp_message.get("text")
            
            if not from_number or not message_text:
                return None
            
            # Crear título de conversación
            title = f"WhatsApp - {from_number}"
            
            # Crear mensaje inicial
            initial_message = Message(
                content=message_text,
                sender="user",
                created_at=datetime.now(timezone.utc)
            )
            
            # Crear conversación
            conversation = Conversation(
                title=title,
                created_at=datetime.now(timezone.utc),
                profile=profile,
                messages=[initial_message]
            )
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation from WhatsApp: {e}")
            return None
    
    async def generate_ai_response(self, conversation: Conversation, user_message: str) -> Optional[str]:
        """
        Genera una respuesta IA para un mensaje de WhatsApp
        """
        try:
            # Aquí puedes integrar con tu servicio de IA
            # Por ahora, retornamos una respuesta simple
            
            profile = conversation.profile
            business_name = profile.business_name or "Asistente"
            
            # Respuesta básica (puedes integrar con OpenAI u otro servicio)
            response = f"Hola, soy {business_name}. He recibido tu mensaje: '{user_message}'. ¿En qué puedo ayudarte?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None
    
    def remove_integration_from_cache(self, profile_id: str):
        """
        Remueve una integración del cache
        """
        keys_to_remove = [key for key in self.active_integrations.keys() if key.startswith(f"{profile_id}_")]
        for key in keys_to_remove:
            del self.active_integrations[key]
            logger.info(f"Removed WhatsApp integration from cache: {key}")
    
    def get_integration_stats(self, profile: Profile) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la integración
        """
        whatsapp_client = self.get_whatsapp_integration(profile)
        if not whatsapp_client:
            return {"status": "inactive", "error": "Integration not found"}
        
        return {
            "status": "active",
            "phone_number_id": whatsapp_client.phone_number_id,
            "api_url": whatsapp_client.api_url,
            "integration_id": profile.id
        }
