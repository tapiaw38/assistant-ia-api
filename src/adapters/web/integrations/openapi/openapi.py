from src.core.platform.config.service import ConfigurationService
from openai import AsyncOpenAI
from typing import Optional
from src.adapters.web.integrations.openapi.context import CONTEXT_INFORMATION
from src.core.domain.model import Profile
import pandas as pd
import requests
from io import BytesIO
from PyPDF2 import PdfReader

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

        extracted_text = ""
        if profile.files:
            for file in profile.files:
                if file.url.endswith(('.xlsx', '.xls')):
                    extracted_text += self._extract_text_from_excel(file.url) + "\n"
                elif file.url.endswith('.pdf'):
                    extracted_text += self._extract_text_from_pdf(file.url) + "\n"
                else:
                    raise ValueError(f"Unsupported file format: {file.url}")

        full_context = f"{context or ''}\n{extracted_text.strip()}"

        print("full_context", full_context)

        question_message = CONTEXT_INFORMATION.format(
            assistant_name=profile.assistant_name,
            business_name=profile.business_name,
            business_context=profile.business_context,
            functions=profile.functions,
            product_context=full_context,
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

    def _extract_text_from_excel(self, file_path: str) -> str:
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            text = ""
            for name, sheet in df.items():
                text += f"\n--- Hoja: {name} ---\n"
                text += sheet.to_string(index=False)
            return text
        except Exception as e:
            return f"Error leyendo archivo Excel {file_path}: {e}"

    def _extract_text_from_pdf(self, file_url: str) -> str:
        try:
            response = requests.get(file_url)
            response.raise_for_status()

            pdf_data = BytesIO(response.content)
            reader = PdfReader(pdf_data)

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ''
            return text
        except Exception as e:
            return f"Error leyendo archivo PDF {file_url}: {e}"
