from fastapi import APIRouter, Request, HTTPException, status, Depends
from src.adapters.web.integrations.integrations import Integrations
from src.adapters.services.services import Services
from src.schemas.schemas import WhatsAppMessageInput, WhatsAppTemplateMessageInput, WhatsAppWebhookPayload
from src.core.domain.model import Profile
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/whatsapp",
    tags=["whatsapp"],
)

def get_services() -> Services:
    return Services.get_instance()

def get_integrations() -> Integrations:
    return Integrations.get_instance()

async def get_user_whatsapp_integration(user_id: str, services: Services, integrations: Integrations):
    """
    Obtiene la integración de WhatsApp del perfil del usuario
    """
    profile = await services.profile.find_by_user_id(user_id)
    
    if not profile or not profile.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    if not profile.data.integrations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No integrations found for this profile"
        )
    
    whatsapp_integration = None
    for integration in profile.data.integrations:
        if integration.type == "whatsapp":
            whatsapp_integration = integration
            break
    
    if not whatsapp_integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="WhatsApp integration not found for this profile"
        )
    
    # Validar configuración
    if not integrations.validate_whatsapp_config(whatsapp_integration.config):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="WhatsApp integration has invalid configuration"
        )
    
    whatsapp_client = integrations.get_whatsapp_integration(whatsapp_integration)
    if not whatsapp_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize WhatsApp client"
        )
    
    return whatsapp_client, profile.data

@router.get("/test-integration")
async def test_integration(
    request: Request,
    services: Services = Depends(get_services),
    integrations: Integrations = Depends(get_integrations)
):
    """
    Endpoint para probar la integración de WhatsApp
    """
    try:
        # Obtener el user_id del request
        user_id = request.state.user.get("user_id")
        
        # Obtener la integración de WhatsApp
        whatsapp_client, profile = await get_user_whatsapp_integration(user_id, services, integrations)
        
        return {
            "status": "integration_active",
            "profile_id": profile.id,
            "whatsapp_phone_id": whatsapp_client.phone_number_id,
            "message": "WhatsApp integration is working correctly"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing WhatsApp integration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error testing integration"
        )

@router.post("/send-message")
async def send_message(
    message_data: WhatsAppMessageInput,
    request: Request,
    services: Services = Depends(get_services),
    integrations: Integrations = Depends(get_integrations)
):
    """
    Endpoint para enviar mensajes de texto a WhatsApp
    """
    try:
        # Obtener el user_id del request (desde el middleware de autorización)
        user_id = request.state.user.get("user_id")
        
        # Obtener la integración de WhatsApp
        whatsapp_client, profile = await get_user_whatsapp_integration(user_id, services, integrations)
        
        # Enviar el mensaje
        result = whatsapp_client.send_message(
            recipient=message_data.recipient,
            message=message_data.message,
            message_type=message_data.message_type
        )
        
        if result["success"]:
            return {
                "status": "message_sent",
                "message_id": result["message_id"],
                "recipient": message_data.recipient
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to send message: {result['error']}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error sending message"
        )

@router.post("/send-template")
async def send_template_message(
    template_data: WhatsAppTemplateMessageInput,
    request: Request,
    services: Services = Depends(get_services),
    integrations: Integrations = Depends(get_integrations)
):
    """
    Endpoint para enviar mensajes de plantilla a WhatsApp
    """
    try:
        # Obtener el user_id del request
        user_id = request.state.user.get("user_id")
        
        # Obtener la integración de WhatsApp
        whatsapp_client, profile = await get_user_whatsapp_integration(user_id, services, integrations)
        
        # Enviar el mensaje de plantilla
        result = whatsapp_client.send_template_message(
            recipient=template_data.recipient,
            template_name=template_data.template_name,
            language_code=template_data.language_code,
            parameters=template_data.parameters
        )
        
        if result["success"]:
            return {
                "status": "template_sent",
                "message_id": result["message_id"],
                "recipient": template_data.recipient
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to send template: {result['error']}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending WhatsApp template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error sending template"
        )
