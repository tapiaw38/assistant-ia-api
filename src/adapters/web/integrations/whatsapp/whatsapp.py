import requests

class WhatsAppIntegration:
    def __init__(self, access_token: str, phone_number_id: str):
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_url = f"https://graph.facebook.com/v18.0/{self.phone_number_id}/messages"

    def send_message(self, recipient: str, message: str) -> bool:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "text",
            "text": {"body": message}
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        return response.status_code == 200

    def receive_message(self, payload):
        messages = []
        try:
            entry = payload.get("entry", [])
            for ent in entry:
                changes = ent.get("changes", [])
            for change in changes:
                value = change.get("value", {})
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
                    messages.append(message_data)
        except Exception as e:
            raise ValueError(f"Error processing WhatsApp message payload: {e}")
        return messages