from src.core.platform.config.service import ConfigurationService
from openai import AsyncOpenAI
from typing import Optional, List
from src.adapters.web.integrations.openapi.context import CONTEXT_INFORMATION
from src.core.domain.model import Profile
from src.core.platform.fileprocessor.pdf_processor import FileImageProcessor
from src.core.platform.fileprocessor.xlsx_processor import ExcelImageProcessor
from src.core.platform.fileprocessor.processor import ImageSearchResult
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

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.image_processor = FileImageProcessor(config_service)
        self.excel_processor = ExcelImageProcessor(config_service)

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
                if file.url.endswith((".xlsx", ".xls")):
                    extracted_text += self._extract_text_from_excel(file.url) + "\n"
                elif file.url.endswith(".pdf"):
                    extracted_text += self._extract_text_from_pdf(file.url) + "\n"
                else:
                    raise ValueError(f"Unsupported file format: {file.url}")

        full_context = f"{context or ''}\n{extracted_text.strip()}"

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
            ],
        )
        return response.choices[0].message.content

    async def search_images_in_files(
        self, search_term: str, profile: Profile, max_results: int = 5
    ) -> List[ImageSearchResult]:
        if not profile.files:
            return []

        all_results = []

        for file in profile.files:
            if file.url.endswith(".pdf"):
                try:
                    results = await self.image_processor.search_images_in_pdf(
                        file.url, search_term, max_results
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error searching images in PDF {file.url}: {e}")
                    continue
            elif file.url.endswith((".xlsx", ".xls")):
                try:
                    results = await self.excel_processor.search_images_in_excel(
                        file.url, search_term, max_results
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error searching images in Excel {file.url}: {e}")
                    continue

        all_results.sort(key=lambda x: x.confidence, reverse=True)
        return all_results[:max_results]

    async def describe_image_from_base64(self, image_base64: str) -> str:
        return await self.image_processor.describe_image(image_base64)

    async def analyze_excel_file(self, excel_url: str) -> dict:
        """Analyze Excel file structure and data"""
        return await self.excel_processor.analyze_excel_data(excel_url)

    def _extract_text_from_excel(self, file_path: str) -> str:
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            text = ""
            for name, sheet in df.items():
                text += f"\n--- Sheet: {name} ---\n"
                text += sheet.to_string(index=False)
            return text
        except Exception as e:
            return f"Error reading Excel file {file_path}: {e}"

    def _extract_text_from_pdf(self, file_url: str) -> str:
        try:
            response = requests.get(file_url)
            response.raise_for_status()

            pdf_data = BytesIO(response.content)
            reader = PdfReader(pdf_data)

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error reading PDF file {file_url}: {e}"
