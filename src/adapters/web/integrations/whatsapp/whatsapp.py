import requests
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class WhatsAppIntegration:
    def __init__(self, access_token: str, phone_number_id: str):
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}/messages"
        self.media_url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}/media"

    def send_message(self, recipient: str, message: str, message_type: str = "text") -> Dict[str, Any]:
        """
        Envía un mensaje de texto a WhatsApp
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": message_type,
        }
        
        if message_type == "text":
            data["text"] = {"body": message}
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Message sent successfully to {recipient}: {result}")
            return {
                "success": True,
                "message_id": result.get("messages", [{}])[0].get("id"),
                "response": result
            }
        except requests.RequestException as e:
            logger.error(f"Error sending message to {recipient}: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": getattr(e, 'response', None)
            }

    def send_template_message(self, recipient: str, template_name: str, language_code: str = "es", parameters: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Envía un mensaje de plantilla a WhatsApp
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        template_data = {
            "name": template_name,
            "language": {"code": language_code}
        }
        
        if parameters:
            template_data["components"] = [{
                "type": "body",
                "parameters": [{"type": "text", "text": param} for param in parameters]
            }]
        
        data = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "template",
            "template": template_data
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Template message sent successfully to {recipient}: {result}")
            return {
                "success": True,
                "message_id": result.get("messages", [{}])[0].get("id"),
                "response": result
            }
        except requests.RequestException as e:
            logger.error(f"Error sending template message to {recipient}: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": getattr(e, 'response', None)
            }

    def receive_message(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Procesa los mensajes recibidos del webhook de WhatsApp
        """
        messages = []
        try:
            entry = payload.get("entry", [])
            for ent in entry:
                changes = ent.get("changes", [])
                for change in changes:
                    value = change.get("value", {})
                    
                    # Procesar mensajes
                    messages_list = value.get("messages", [])
                    for msg in messages_list:
                        message_data = {
                            "from": msg.get("from"),
                            "id": msg.get("id"),
                            "timestamp": msg.get("timestamp"),
                            "type": msg.get("type"),
                            "text": msg.get("text", {}).get("body") if msg.get("type") == "text" else None,
                            "raw": msg
                        }
                        
                        # Agregar información adicional según el tipo de mensaje
                        if msg.get("type") == "image":
                            message_data["image"] = msg.get("image", {})
                        elif msg.get("type") == "document":
                            message_data["document"] = msg.get("document", {})
                        elif msg.get("type") == "audio":
                            message_data["audio"] = msg.get("audio", {})
                        elif msg.get("type") == "location":
                            message_data["location"] = msg.get("location", {})
                        
                        messages.append(message_data)
                    
                    # Procesar status de mensajes (entregados, leídos, etc.)
                    statuses = value.get("statuses", [])
                    for status in statuses:
                        status_data = {
                            "id": status.get("id"),
                            "status": status.get("status"),
                            "timestamp": status.get("timestamp"),
                            "recipient_id": status.get("recipient_id"),
                            "type": "status_update",
                            "raw": status
                        }
                        messages.append(status_data)
                        
        except Exception as e:
            logger.error(f"Error processing WhatsApp message payload: {e}")
            raise ValueError(f"Error processing WhatsApp message payload: {e}")
        
        return messages

    def mark_message_as_read(self, message_id: str) -> Dict[str, Any]:
        """
        Marca un mensaje como leído
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Message {message_id} marked as read: {result}")
            return {
                "success": True,
                "response": result
            }
        except requests.RequestException as e:
            logger.error(f"Error marking message as read: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_media_url(self, media_id: str) -> Optional[str]:
        """
        Obtiene la URL de un archivo multimedia
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
        }
        
        try:
            response = requests.get(f"https://graph.facebook.com/v18.0/{media_id}", headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("url")
        except requests.RequestException as e:
            logger.error(f"Error getting media URL: {e}")
            return None

    def download_media(self, media_url: str) -> Optional[bytes]:
        """
        Descarga un archivo multimedia
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
        }
        
        try:
            response = requests.get(media_url, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"Error downloading media: {e}")
            return None

    @staticmethod
    def validate_webhook_signature(payload: str, signature: str, app_secret: str) -> bool:
        """
        Valida la firma del webhook para asegurar que proviene de WhatsApp
        """
        import hmac
        import hashlib
        
        expected_signature = hmac.new(
            app_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected_signature}", signature)