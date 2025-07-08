# Integración WhatsApp Business

Esta documentación describe cómo configurar y usar la integración de WhatsApp Business con el asistente IA.

## Configuración inicial

### 1. Crear la integración

Para crear una integración de WhatsApp, usa el endpoint existente:

```bash
POST /profile/integrations
```

```json
{
  "name": "WhatsApp Business",
  "type": "whatsapp",
  "config": {
    "access_token": "YOUR_WHATSAPP_ACCESS_TOKEN",
    "phone_number_id": "YOUR_PHONE_NUMBER_ID",
    "webhook_verify_token": "YOUR_WEBHOOK_VERIFY_TOKEN",
    "app_secret": "YOUR_APP_SECRET"
  }
}
```

### 2. Configurar el webhook

En tu aplicación de WhatsApp Business en Facebook for Developers:

1. Ve a WhatsApp > Configuración
2. Configura el webhook:
   - **URL**: `https://tu-dominio.com/webhooks/whatsapp/receive`
   - **Token de verificación**: El valor de `webhook_verify_token`
   - **Campos**: `messages`, `message_deliveries`, `message_reads`

## Endpoints disponibles

### Enviar mensajes de texto

```bash
POST /whatsapp/send-message
```

```json
{
  "recipient": "521234567890",
  "message": "¡Hola! Este es un mensaje de prueba.",
  "message_type": "text"
}
```

### Enviar mensajes de plantilla

```bash
POST /whatsapp/send-template
```

```json
{
  "recipient": "521234567890",
  "template_name": "hello_world",
  "language_code": "es",
  "parameters": ["Nombre del usuario"]
}
```

### Probar integración

```bash
GET /whatsapp/test-integration
```

### Webhook de verificación

```bash
GET /webhooks/whatsapp/verify?hub.mode=subscribe&hub.verify_token=YOUR_TOKEN&hub.challenge=CHALLENGE
```

### Recibir mensajes

```bash
POST /webhooks/whatsapp/receive
```

## Funcionalidades

### Envío de mensajes

- ✅ Mensajes de texto
- ✅ Mensajes de plantilla
- ✅ Respuestas automáticas
- ✅ Validación de configuración
- ✅ Manejo de errores

### Recepción de mensajes

- ✅ Procesamiento de webhooks
- ✅ Extracción de mensajes de texto
- ✅ Soporte para múltiples tipos de mensaje
- ✅ Validación de firmas (opcional)
- ✅ Marca de leído automática

### Tipos de mensaje soportados

- **Texto**: Mensajes de texto plano
- **Plantillas**: Mensajes pre-aprobados por WhatsApp
- **Imágenes**: Recepción de imágenes (procesamiento básico)
- **Documentos**: Recepción de documentos
- **Audio**: Recepción de mensajes de voz
- **Ubicación**: Recepción de ubicaciones

## Configuración de campos requeridos

### access_token

Token de acceso de WhatsApp Business API. Se obtiene desde Facebook for Developers.

### phone_number_id

ID del número de teléfono configurado en WhatsApp Business. Se encuentra en la configuración del número de teléfono.

### webhook_verify_token

Token usado para verificar el webhook. Debe coincidir con el configurado en Facebook for Developers.

### app_secret (opcional)

Secreto de la aplicación para validar firmas de webhook. Recomendado para producción.

## Ejemplo de flujo completo

1. **Configurar integración**:

   ```bash
   curl -X POST https://tu-api.com/profile/integrations \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "WhatsApp Business",
       "type": "whatsapp",
       "config": {
         "access_token": "YOUR_ACCESS_TOKEN",
         "phone_number_id": "YOUR_PHONE_ID",
         "webhook_verify_token": "YOUR_VERIFY_TOKEN"
       }
     }'
   ```

2. **Probar integración**:

   ```bash
   curl -X GET https://tu-api.com/whatsapp/test-integration \
     -H "Authorization: Bearer YOUR_JWT_TOKEN"
   ```

3. **Enviar mensaje**:
   ```bash
   curl -X POST https://tu-api.com/whatsapp/send-message \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "recipient": "521234567890",
       "message": "¡Hola desde el asistente IA!"
     }'
   ```

## Errores comunes

### Error 403: Invalid verification token

- **Causa**: El token de verificación no coincide
- **Solución**: Verifica que `webhook_verify_token` sea correcto

### Error 400: Invalid phone number format

- **Causa**: Formato de número de teléfono incorrecto
- **Solución**: Usa formato internacional (ej: 521234567890)

### Error 401: Invalid access token

- **Causa**: Access token inválido o expirado
- **Solución**: Genera un nuevo access token en Facebook for Developers

### Error 500: Failed to send message

- **Causa**: Problema con la API de WhatsApp
- **Solución**: Verifica que el número de teléfono esté registrado y la configuración sea correcta

## Mejores prácticas

1. **Validar números de teléfono** antes de enviar mensajes
2. **Usar plantillas aprobadas** para mensajes promocionales
3. **Implementar rate limiting** para evitar spam
4. **Validar firmas de webhook** en producción
5. **Manejar errores gracefully** y proporcionar feedback al usuario
6. **Almacenar logs** de mensajes enviados y recibidos
7. **Cumplir con las políticas** de WhatsApp Business

## Próximas mejoras

- [ ] Soporte para mensajes multimedia
- [ ] Gestión de conversaciones
- [ ] Respuestas automáticas inteligentes
- [ ] Integración con el asistente IA
- [ ] Dashboard de métricas
- [ ] Plantillas dinámicas
- [ ] Soporte para múltiples idiomas
