from src.core.platform.config.service import ConfigurationService
from openai import AsyncOpenAI
from typing import Optional
from src.adapters.integrations.openapi.context import CONTEXT_INFORMATION
from src.core.domain.model import Profile

class OpenAIIntegration:
    def __init__(self, config_service: ConfigurationService):
        config = config_service.openai_config

        self.api_key = config.api_key
        self.base_url = config.base_url
        self.model = config.model
        self.role = config.role

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def ask(
            self, 
            question: str, 
            context: Optional[str], 
            previous_messages: str,
            profile: Profile,
        ) -> str:
        question_message = CONTEXT_INFORMATION.format(
            assistant_name=profile.assistant_name,
            business_name=profile.business_name,
            business_context=profile.business_context,
            functions=profile.functions,
            product_context=context,
            question=question,
            previous_messages=previous_messages,
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": self.role, "content": question_message},
            ]
        )
        return response.choices[0].message.content
