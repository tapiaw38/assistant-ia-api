from fastapi import APIRouter, Request, HTTPException, status, Query
from src.adapters.web.integrations.whatsapp.whatsapp import WhatsAppIntegration
from src.adapters.services.services import Services
from src.schemas.schemas import WhatsAppWebhookPayload
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/webhooks/whatsapp",
    tags=["webhooks", "whatsapp"],
)

@router.get("/verify")
async def verify_webhook(
    request: Request,
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge")
):
    """
    Endpoint para verificar el webhook de WhatsApp
    """
    try:
        # Obtener el token de verificación desde la configuración
        # En un entorno real, esto debería venir de una variable de entorno
        VERIFY_TOKEN = "your_verify_token_here"  # Cambiar por tu token real
        
        if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
            logger.info(f"Webhook verified successfully. Challenge: {hub_challenge}")
            return int(hub_challenge)
        else:
            logger.warning(f"Invalid verification attempt. Mode: {hub_mode}, Token: {hub_verify_token}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid verification token"
            )
    
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid challenge parameter"
        )
    except Exception as e:
        logger.error(f"Error verifying webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error verifying webhook"
        )

@router.post("/receive")
async def receive_webhook(request: Request):
    """
    Endpoint para recibir mensajes de WhatsApp
    """
    try:
        # Obtener el payload del webhook
        payload = await request.json()
        logger.info(f"Received WhatsApp webhook: {payload}")
        
        # Validar la firma del webhook (opcional pero recomendado)
        signature = request.headers.get("X-Hub-Signature-256")
        if signature:
            # Aquí deberías validar la firma usando tu app_secret
            # WhatsAppIntegration.validate_webhook_signature(payload_str, signature, app_secret)
            pass
        
        # Procesar el mensaje usando la clase WhatsAppIntegration
        # Crear una instancia temporal para procesar el mensaje
        temp_whatsapp = WhatsAppIntegration(
            access_token="temp_token",
            phone_number_id="temp_phone_id"
        )
        
        messages = temp_whatsapp.receive_message(payload)
        
        if not messages:
            return {"status": "no_messages_processed"}
        
        # Procesar cada mensaje
        services = Services.get_instance()
        processed_messages = []
        
        for message in messages:
            try:
                # Aquí puedes implementar la lógica para:
                # 1. Identificar el usuario/perfil basado en el número de teléfono
                # 2. Crear o continuar una conversación
                # 3. Generar una respuesta usando el asistente IA
                # 4. Enviar la respuesta de vuelta a WhatsApp
                
                if message.get("type") == "status_update":
                    # Manejar actualizaciones de estado
                    logger.info(f"Message status update: {message['status']} for message {message['id']}")
                    continue
                
                from_number = message.get("from")
                message_text = message.get("text")
                message_type = message.get("type")
                
                if not from_number:
                    logger.warning(f"Message without sender: {message}")
                    continue
                
                logger.info(f"Processing message from {from_number}: {message_text} (type: {message_type})")
                
                # Aquí puedes buscar el perfil del usuario basado en el número de teléfono
                # Por ahora, simplemente registramos el mensaje
                
                processed_messages.append({
                    "from": from_number,
                    "text": message_text,
                    "type": message_type,
                    "timestamp": message.get("timestamp"),
                    "id": message.get("id")
                })
                
                # Opcional: marcar el mensaje como leído
                # if message.get("id"):
                #     temp_whatsapp.mark_message_as_read(message["id"])
                
            except Exception as e:
                logger.error(f"Error processing individual message: {e}")
                continue
        
        return {
            "status": "messages_processed",
            "count": len(processed_messages),
            "messages": processed_messages
        }
    
    except Exception as e:
        logger.error(f"Error processing WhatsApp webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )

@router.post("/test-webhook")
async def test_webhook(webhook_data: WhatsAppWebhookPayload):
    """
    Endpoint para probar el procesamiento de webhooks
    """
    try:
        # Crear una instancia temporal para procesar el mensaje
        temp_whatsapp = WhatsAppIntegration(
            access_token="temp_token",
            phone_number_id="temp_phone_id"
        )
        
        # Convertir el payload a dict
        payload_dict = webhook_data.dict()
        
        # Procesar el mensaje
        messages = temp_whatsapp.receive_message(payload_dict)
        
        return {
            "status": "test_successful",
            "messages_found": len(messages),
            "messages": messages
        }
    
    except Exception as e:
        logger.error(f"Error testing webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing webhook: {str(e)}"
        )
