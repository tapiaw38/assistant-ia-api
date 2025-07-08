import pytest
from unittest.mock import Mock, patch, MagicMock
from src.adapters.web.integrations.whatsapp.whatsapp import WhatsAppIntegration
from src.adapters.services.whatsapp.service import WhatsAppService
from src.core.domain.model import Profile, Integration
from src.schemas.schemas import WhatsAppMessageInput, WhatsAppTemplateMessageInput
from datetime import datetime, timezone


class TestWhatsAppIntegration:
    """
    Tests para la integración de WhatsApp
    """
    
    def setup_method(self):
        """
        Configuración para cada test
        """
        self.access_token = "test_access_token"
        self.phone_number_id = "test_phone_number_id"
        self.whatsapp = WhatsAppIntegration(self.access_token, self.phone_number_id)
    
    def test_init_whatsapp_integration(self):
        """
        Test inicialización de WhatsAppIntegration
        """
        assert self.whatsapp.access_token == self.access_token
        assert self.whatsapp.phone_number_id == self.phone_number_id
        assert "graph.facebook.com" in self.whatsapp.api_url
        assert self.phone_number_id in self.whatsapp.api_url
    
    @patch('src.adapters.web.integrations.whatsapp.whatsapp.requests.post')
    def test_send_message_success(self, mock_post):
        """
        Test envío exitoso de mensaje
        """
        # Configurar mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [{"id": "test_message_id"}]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Ejecutar
        result = self.whatsapp.send_message("1234567890", "Test message")
        
        # Verificar
        assert result["success"] is True
        assert result["message_id"] == "test_message_id"
        mock_post.assert_called_once()
    
    @patch('src.adapters.web.integrations.whatsapp.whatsapp.requests.post')
    def test_send_message_failure(self, mock_post):
        """
        Test fallo en envío de mensaje
        """
        # Configurar mock para fallo
        mock_post.side_effect = Exception("Network error")
        
        # Ejecutar
        result = self.whatsapp.send_message("1234567890", "Test message")
        
        # Verificar
        assert result["success"] is False
        assert "error" in result
    
    def test_receive_message_valid_payload(self):
        """
        Test procesamiento de mensaje válido
        """
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "1234567890",
                                        "id": "test_message_id",
                                        "timestamp": "1234567890",
                                        "type": "text",
                                        "text": {"body": "Hello World"}
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
        
        messages = self.whatsapp.receive_message(payload)
        
        assert len(messages) == 1
        assert messages[0]["from"] == "1234567890"
        assert messages[0]["text"] == "Hello World"
        assert messages[0]["type"] == "text"
    
    def test_receive_message_empty_payload(self):
        """
        Test procesamiento de payload vacío
        """
        payload = {"entry": []}
        
        messages = self.whatsapp.receive_message(payload)
        
        assert len(messages) == 0
    
    def test_receive_message_invalid_payload(self):
        """
        Test procesamiento de payload inválido
        """
        payload = {"invalid": "payload"}
        
        messages = self.whatsapp.receive_message(payload)
        
        assert len(messages) == 0


class TestWhatsAppService:
    """
    Tests para el servicio de WhatsApp
    """
    
    def setup_method(self):
        """
        Configuración para cada test
        """
        self.service = WhatsAppService()
        self.profile = Profile(
            _id="test_profile_id",
            user_id="test_user_id",
            assistant_name="Test Assistant",
            business_name="Test Business",
            functions="Test Functions",
            business_context="Test Context",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            integrations=[
                Integration(
                    _id="test_integration_id",
                    name="WhatsApp",
                    type="whatsapp",
                    config={
                        "access_token": "test_access_token",
                        "phone_number_id": "test_phone_number_id"
                    },
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
            ]
        )
    
    def test_validate_whatsapp_config_valid(self):
        """
        Test validación de configuración válida
        """
        config = {
            "access_token": "valid_token_123456",
            "phone_number_id": "1234567890"
        }
        
        is_valid, message = self.service.validate_whatsapp_config(config)
        
        assert is_valid is True
        assert "valid" in message.lower()
    
    def test_validate_whatsapp_config_missing_token(self):
        """
        Test validación con token faltante
        """
        config = {
            "phone_number_id": "1234567890"
        }
        
        is_valid, message = self.service.validate_whatsapp_config(config)
        
        assert is_valid is False
        assert "access_token" in message
    
    def test_validate_whatsapp_config_invalid_phone_id(self):
        """
        Test validación con phone_number_id inválido
        """
        config = {
            "access_token": "valid_token_123456",
            "phone_number_id": "invalid_phone_id"
        }
        
        is_valid, message = self.service.validate_whatsapp_config(config)
        
        assert is_valid is False
        assert "phone_number_id" in message
    
    def test_get_whatsapp_integration_success(self):
        """
        Test obtención exitosa de integración
        """
        integration = self.service.get_whatsapp_integration(self.profile)
        
        assert integration is not None
        assert integration.phone_number_id == "test_phone_number_id"
    
    def test_get_whatsapp_integration_no_integrations(self):
        """
        Test obtención de integración sin integraciones
        """
        profile_no_integrations = Profile(
            _id="test_profile_id",
            user_id="test_user_id",
            assistant_name="Test Assistant",
            business_name="Test Business",
            functions="Test Functions",
            business_context="Test Context",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            integrations=None
        )
        
        integration = self.service.get_whatsapp_integration(profile_no_integrations)
        
        assert integration is None
    
    @patch('src.adapters.services.whatsapp.service.WhatsAppIntegration.send_message')
    async def test_send_message_success(self, mock_send):
        """
        Test envío exitoso de mensaje a través del servicio
        """
        # Configurar mock
        mock_send.return_value = {
            "success": True,
            "message_id": "test_message_id"
        }
        
        # Crear input
        message_input = WhatsAppMessageInput(
            recipient="1234567890",
            message="Test message",
            message_type="text"
        )
        
        # Ejecutar
        result = await self.service.send_message(self.profile, message_input)
        
        # Verificar
        assert result["success"] is True
        assert result["message_id"] == "test_message_id"
    
    def test_get_integration_stats(self):
        """
        Test obtención de estadísticas de integración
        """
        stats = self.service.get_integration_stats(self.profile)
        
        assert stats["status"] == "active"
        assert stats["phone_number_id"] == "test_phone_number_id"
        assert "api_url" in stats
    
    def test_remove_integration_from_cache(self):
        """
        Test remoción de integración del cache
        """
        # Primero agregar al cache
        self.service.get_whatsapp_integration(self.profile)
        
        # Verificar que está en cache
        assert len(self.service.active_integrations) > 0
        
        # Remover del cache
        self.service.remove_integration_from_cache(self.profile.id)
        
        # Verificar que se removió
        assert len(self.service.active_integrations) == 0


class TestWhatsAppIntegrationEndToEnd:
    """
    Tests end-to-end para la integración completa
    """
    
    def test_webhook_payload_processing(self):
        """
        Test procesamiento completo de webhook
        """
        # Payload de ejemplo de WhatsApp
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
                                "contacts": [
                                    {
                                        "profile": {"name": "John Doe"},
                                        "wa_id": "WHATSAPP_ID"
                                    }
                                ],
                                "messages": [
                                    {
                                        "from": "WHATSAPP_ID",
                                        "id": "wamid.MESSAGE_ID",
                                        "timestamp": "1234567890",
                                        "text": {"body": "Hello, I need help."},
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
        
        # Crear integración
        whatsapp = WhatsAppIntegration("test_token", "test_phone_id")
        
        # Procesar webhook
        messages = whatsapp.receive_message(webhook_payload)
        
        # Verificar resultado
        assert len(messages) == 1
        assert messages[0]["from"] == "WHATSAPP_ID"
        assert messages[0]["text"] == "Hello, I need help."
        assert messages[0]["type"] == "text"
    
    def test_configuration_validation_flow(self):
        """
        Test flujo completo de validación de configuración
        """
        service = WhatsAppService()
        
        # Test configuración válida
        valid_config = {
            "access_token": "EAABsBCS1iHgBAABCD1234567890",
            "phone_number_id": "1234567890123"
        }
        
        is_valid, message = service.validate_whatsapp_config(valid_config)
        assert is_valid is True
        
        # Test configuración inválida
        invalid_config = {
            "access_token": "short",
            "phone_number_id": "not_numeric"
        }
        
        is_valid, message = service.validate_whatsapp_config(invalid_config)
        assert is_valid is False


# Configuración de pytest
@pytest.fixture
def whatsapp_integration():
    return WhatsAppIntegration("test_token", "test_phone_id")


@pytest.fixture
def whatsapp_service():
    return WhatsAppService()


@pytest.fixture
def sample_profile():
    return Profile(
        _id="test_profile_id",
        user_id="test_user_id",
        assistant_name="Test Assistant",
        business_name="Test Business",
        functions="Test Functions",
        business_context="Test Context",
        is_active=True,
        created_at=datetime.now(timezone.utc),
        integrations=[
            Integration(
                _id="test_integration_id",
                name="WhatsApp",
                type="whatsapp",
                config={
                    "access_token": "test_access_token",
                    "phone_number_id": "test_phone_number_id"
                },
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
        ]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
