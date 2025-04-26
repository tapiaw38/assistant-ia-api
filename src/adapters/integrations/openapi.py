from src.core.platform.config.service import ConfigurationService
from openai import AsyncOpenAI
from typing import Optional

CONTEXT_INFORMATION = "" \
"You are a helpful assistant. \
You are given the following context: \
{context} \
You are given the following question: \
{question} \
Answer the question as concisely as possible, using the context provided. " \
"return the answer as a string without any additional text and respond in spanish"

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

    async def ask(self, question: str, context: Optional[str]) -> str:
        question_message = CONTEXT_INFORMATION.format(question=question, context=context)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": self.role, "content": question_message},
            ]
        )
        return response.choices[0].message.content
