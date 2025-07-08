"""
Ejemplo de configuración para integrar WhatsApp Business con el asistente IA

Este archivo muestra ejemplos de cómo configurar y usar la integración de WhatsApp.
"""

# Ejemplo de configuración para crear una integración de WhatsApp
WHATSAPP_INTEGRATION_CONFIG = {
    "name": "WhatsApp Business",
    "type": "whatsapp",
    "config": {
        "access_token": "YOUR_WHATSAPP_ACCESS_TOKEN",
        "phone_number_id": "YOUR_PHONE_NUMBER_ID",
        "webhook_verify_token": "YOUR_WEBHOOK_VERIFY_TOKEN",
        "app_secret": "YOUR_APP_SECRET"
    }
}

# Ejemplo de payload para enviar un mensaje de texto
SEND_TEXT_MESSAGE_PAYLOAD = {
    "recipient": "521234567890",  # Número de teléfono con código de país
    "message": "¡Hola! Este es un mensaje de prueba desde el asistente IA.",
    "message_type": "text"
}

# Ejemplo de payload para enviar un mensaje de plantilla
SEND_TEMPLATE_MESSAGE_PAYLOAD = {
    "recipient": "521234567890",
    "template_name": "hello_world",
    "language_code": "es",
    "parameters": ["Usuario", "Asistente IA"]
}

# Ejemplo de webhook payload que WhatsApp envía
WEBHOOK_PAYLOAD_EXAMPLE = {
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
                        "contacts": [
                            {
                                "profile": {
                                    "name": "John Doe"
                                },
                                "wa_id": "WHATSAPP_ID"
                            }
                        ],
                        "messages": [
                            {
                                "from": "WHATSAPP_ID",
                                "id": "wamid.MESSAGE_ID",
                                "timestamp": "1234567890",
                                "text": {
                                    "body": "Hola, necesito ayuda con mi pedido."
                                },
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

# Ejemplo de configuración de webhook en WhatsApp Business
WEBHOOK_CONFIGURATION_GUIDE = """
Para configurar el webhook en WhatsApp Business:

1. Ve a la configuración de tu aplicación en Facebook for Developers
2. Selecciona "WhatsApp" > "Configuración"
3. En la sección "Webhooks", agrega:
   - URL de callback: https://tu-dominio.com/webhooks/whatsapp/receive
   - Token de verificación: tu_webhook_verify_token
   - Campos de suscripción: messages, message_deliveries, message_reads

4. Verifica el webhook usando: https://tu-dominio.com/webhooks/whatsapp/verify

5. Guarda los siguientes datos en tu integración:
   - access_token: Token de acceso de WhatsApp Business
   - phone_number_id: ID del número de teléfono
   - webhook_verify_token: Token de verificación del webhook
   - app_secret: Secreto de la aplicación (para validar firmas)
"""

# Ejemplo de uso de la API
API_USAGE_EXAMPLES = {
    "crear_integracion": {
        "method": "POST",
        "url": "/profile/integrations",
        "headers": {
            "Authorization": "Bearer YOUR_JWT_TOKEN",
            "Content-Type": "application/json"
        },
        "body": WHATSAPP_INTEGRATION_CONFIG
    },
    "enviar_mensaje": {
        "method": "POST",
        "url": "/whatsapp/send-message",
        "headers": {
            "Authorization": "Bearer YOUR_JWT_TOKEN",
            "Content-Type": "application/json"
        },
        "body": SEND_TEXT_MESSAGE_PAYLOAD
    },
    "enviar_plantilla": {
        "method": "POST",
        "url": "/whatsapp/send-template",
        "headers": {
            "Authorization": "Bearer YOUR_JWT_TOKEN",
            "Content-Type": "application/json"
        },
        "body": SEND_TEMPLATE_MESSAGE_PAYLOAD
    },
    "probar_integracion": {
        "method": "GET",
        "url": "/whatsapp/test-integration",
        "headers": {
            "Authorization": "Bearer YOUR_JWT_TOKEN"
        }
    }
}

# Validaciones recomendadas
VALIDATION_CHECKLIST = [
    "✓ Acceso token de WhatsApp Business válido",
    "✓ Phone Number ID correcto",
    "✓ Webhook configurado y verificado",
    "✓ Certificado SSL válido para el webhook",
    "✓ Plantillas de mensajes aprobadas por WhatsApp",
    "✓ Números de teléfono en formato internacional",
    "✓ Cumplimiento con políticas de WhatsApp Business"
]

# Errores comunes y soluciones
COMMON_ERRORS = {
    "invalid_access_token": {
        "error": "Invalid access token",
        "solution": "Verifica que el access token sea válido y no haya expirado"
    },
    "invalid_phone_number": {
        "error": "Invalid phone number format",
        "solution": "Usa formato internacional (ej: 521234567890)"
    },
    "webhook_verification_failed": {
        "error": "Webhook verification failed",
        "solution": "Verifica que el verify_token coincida con el configurado"
    },
    "message_template_not_approved": {
        "error": "Message template not approved",
        "solution": "Usa solo plantillas aprobadas por WhatsApp Business"
    }
}
