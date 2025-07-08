#!/usr/bin/env python3
"""
Script de ejemplo para configurar y usar la integración de WhatsApp Business

Este script muestra cómo:
1. Crear una integración de WhatsApp
2. Enviar mensajes
3. Procesar webhooks
4. Validar configuración
"""

import json
import requests
from typing import Dict, Any


class WhatsAppIntegrationExample:
    """
    Ejemplo de uso de la integración de WhatsApp
    """
    
    def __init__(self, api_base_url: str, jwt_token: str):
        self.api_base_url = api_base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json"
        }
    
    def create_whatsapp_integration(self, whatsapp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea una integración de WhatsApp
        """
        integration_data = {
            "name": "WhatsApp Business",
            "type": "whatsapp",
            "config": whatsapp_config
        }
        
        response = requests.post(
            f"{self.api_base_url}/profile/integrations",
            headers=self.headers,
            json=integration_data
        )
        
        if response.status_code == 201:
            print("✅ Integración de WhatsApp creada exitosamente")
            return response.json()
        else:
            print(f"❌ Error creando integración: {response.text}")
            return {"error": response.text}
    
    def send_text_message(self, recipient: str, message: str) -> Dict[str, Any]:
        """
        Envía un mensaje de texto
        """
        message_data = {
            "recipient": recipient,
            "message": message,
            "message_type": "text"
        }
        
        response = requests.post(
            f"{self.api_base_url}/whatsapp/send-message",
            headers=self.headers,
            json=message_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Mensaje enviado a {recipient}: {result.get('message_id')}")
            return result
        else:
            print(f"❌ Error enviando mensaje: {response.text}")
            return {"error": response.text}
    
    def send_template_message(self, recipient: str, template_name: str, parameters: list = None) -> Dict[str, Any]:
        """
        Envía un mensaje de plantilla
        """
        template_data = {
            "recipient": recipient,
            "template_name": template_name,
            "language_code": "es",
            "parameters": parameters or []
        }
        
        response = requests.post(
            f"{self.api_base_url}/whatsapp/send-template",
            headers=self.headers,
            json=template_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Plantilla enviada a {recipient}: {result.get('message_id')}")
            return result
        else:
            print(f"❌ Error enviando plantilla: {response.text}")
            return {"error": response.text}
    
    def test_integration(self) -> Dict[str, Any]:
        """
        Prueba la integración
        """
        response = requests.get(
            f"{self.api_base_url}/whatsapp/test-integration",
            headers=self.headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Integración activa: {result.get('message')}")
            return result
        else:
            print(f"❌ Error probando integración: {response.text}")
            return {"error": response.text}
    
    def simulate_webhook(self, webhook_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simula un webhook para pruebas
        """
        response = requests.post(
            f"{self.api_base_url}/webhooks/whatsapp/test-webhook",
            json=webhook_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Webhook procesado: {result.get('messages_found')} mensajes encontrados")
            return result
        else:
            print(f"❌ Error procesando webhook: {response.text}")
            return {"error": response.text}


def main():
    """
    Función principal para ejecutar ejemplos
    """
    # Configuración (cambiar por tus valores reales)
    API_BASE_URL = "http://localhost:8000"  # URL de tu API
    JWT_TOKEN = "your_jwt_token_here"  # Tu token JWT
    
    # Configuración de WhatsApp (cambiar por tus valores reales)
    WHATSAPP_CONFIG = {
        "access_token": "YOUR_WHATSAPP_ACCESS_TOKEN",
        "phone_number_id": "YOUR_PHONE_NUMBER_ID",
        "webhook_verify_token": "YOUR_WEBHOOK_VERIFY_TOKEN",
        "app_secret": "YOUR_APP_SECRET"
    }
    
    # Crear instancia del ejemplo
    example = WhatsAppIntegrationExample(API_BASE_URL, JWT_TOKEN)
    
    print("🚀 Iniciando ejemplo de integración WhatsApp")
    print("=" * 50)
    
    # 1. Crear integración
    print("\n1. Creando integración de WhatsApp...")
    integration_result = example.create_whatsapp_integration(WHATSAPP_CONFIG)
    
    if "error" in integration_result:
        print("❌ No se pudo crear la integración. Abortando.")
        return
    
    # 2. Probar integración
    print("\n2. Probando integración...")
    test_result = example.test_integration()
    
    if "error" in test_result:
        print("❌ La integración no está funcionando correctamente.")
        return
    
    # 3. Enviar mensaje de texto
    print("\n3. Enviando mensaje de texto...")
    recipient = "521234567890"  # Cambiar por un número real
    message = "¡Hola! Este es un mensaje de prueba desde el asistente IA."
    
    send_result = example.send_text_message(recipient, message)
    
    # 4. Enviar mensaje de plantilla
    print("\n4. Enviando mensaje de plantilla...")
    template_result = example.send_template_message(
        recipient,
        "hello_world",
        ["Usuario de prueba"]
    )
    
    # 5. Simular webhook
    print("\n5. Simulando webhook...")
    webhook_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550559999",
                                "phone_number_id": "PHONE_NUMBER_ID"
                            },
                            "messages": [
                                {
                                    "from": "WHATSAPP_ID",
                                    "id": "wamid.MESSAGE_ID",
                                    "timestamp": "1234567890",
                                    "text": {"body": "Hola, necesito ayuda."},
                                    "type": "text"
                                }
                            ]
                        },
                        "field": "messages"
                    }
                ]
            }
        ]
    }
    
    webhook_result = example.simulate_webhook(webhook_payload)
    
    print("\n✅ Ejemplo completado exitosamente!")
    print("=" * 50)
    
    # Mostrar resumen
    print("\n📊 Resumen de resultados:")
    print(f"- Integración creada: {'✅' if 'error' not in integration_result else '❌'}")
    print(f"- Integración funcionando: {'✅' if 'error' not in test_result else '❌'}")
    print(f"- Mensaje enviado: {'✅' if 'error' not in send_result else '❌'}")
    print(f"- Plantilla enviada: {'✅' if 'error' not in template_result else '❌'}")
    print(f"- Webhook procesado: {'✅' if 'error' not in webhook_result else '❌'}")


def show_configuration_guide():
    """
    Muestra una guía de configuración
    """
    print("""
🔧 GUÍA DE CONFIGURACIÓN DE WHATSAPP BUSINESS

1. Crear aplicación en Facebook for Developers:
   - Ve a https://developers.facebook.com/
   - Crea una nueva aplicación
   - Agrega el producto "WhatsApp Business"

2. Configurar WhatsApp Business:
   - Selecciona tu número de teléfono
   - Copia el "Phone Number ID"
   - Genera un "Access Token"

3. Configurar Webhook:
   - URL: https://tu-dominio.com/webhooks/whatsapp/receive
   - Verify Token: Define un token personalizado
   - Campos: messages, message_deliveries, message_reads

4. Configurar variables en el script:
   - API_BASE_URL: URL de tu API
   - JWT_TOKEN: Token de autenticación
   - WHATSAPP_CONFIG: Configuración de WhatsApp

5. Ejecutar el script:
   python whatsapp_example.py

📋 CHECKLIST ANTES DE USAR:
   □ Aplicación de Facebook configurada
   □ Access Token válido
   □ Phone Number ID correcto
   □ Webhook configurado y verificado
   □ SSL certificado en tu dominio
   □ Plantillas de mensaje aprobadas
   □ Números de teléfono en formato internacional
   □ Políticas de WhatsApp Business cumplidas
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_configuration_guide()
    else:
        main()
