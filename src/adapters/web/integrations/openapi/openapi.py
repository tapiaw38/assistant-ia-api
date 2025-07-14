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
import asyncio
import aiohttp
import aiofiles
import tempfile
import os
from pathlib import Path


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
                try:
                    if file.url.lower().endswith((".xlsx", ".xls")):
                        extracted_text += (
                            await self._extract_text_from_excel_async(file.url) + "\n"
                        )
                    elif file.url.lower().endswith(".pdf"):
                        extracted_text += (
                            await self._extract_text_from_pdf_async(file.url) + "\n"
                        )
                    else:
                        raise ValueError(f"Unsupported file format: {file.url}")
                except Exception as e:
                    print(f"Error processing file {file.url}: {e}")
                    continue

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
            if file.url.lower().endswith(".pdf"):
                try:
                    results = await self.image_processor.search_images_in_pdf(
                        file.url, search_term, max_results
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error searching images in PDF {file.url}: {e}")
                    continue
            elif file.url.lower().endswith((".xlsx", ".xls")):
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

    async def extract_text_and_images_from_pdf(
        self, pdf_url: str, search_term: Optional[str] = None
    ) -> dict:
        """
        Extract text and images from PDF using OCR if necessary.
        Uses the enhanced PDF processor with OCR capabilities.

        Args:
            pdf_url: URL of the PDF to process
            search_term: Optional search term to filter images

        Returns:
            Dictionary with extracted text, related images and metadata
        """
        try:
            return await self.image_processor.extract_text_and_images(pdf_url, search_term)
        except Exception as e:
            return {
                "error": f"Error processing PDF with OCR: {str(e)}",
                "analysis": None,
                "extracted_text": None,
                "related_images": None,
                "recommendations": ["Manual review required"]
            }

    async def analyze_pdf_content(self, pdf_url: str) -> dict:
        """
        Analyze PDF content to determine processing method needed.

        Args:
            pdf_url: URL of the PDF to analyze

        Returns:
            Dictionary with analysis results
        """
        try:
            analysis_result = await self.image_processor.analyze_pdf_content(pdf_url)
            return {
                "has_text": analysis_result.has_text,
                "total_pages": analysis_result.total_pages,
                "text_pages": analysis_result.text_pages,
                "image_pages": analysis_result.image_pages,
                "processing_method": analysis_result.processing_method,
                "text_extraction_count": len(analysis_result.text_extraction_results)
            }
        except Exception as e:
            return {
                "error": f"Error analyzing PDF content: {str(e)}",
                "has_text": False,
                "total_pages": 0,
                "text_pages": [],
                "image_pages": [],
                "processing_method": "error"
            }

    async def search_images_with_text_analysis(
        self, search_term: str, profile: Profile, max_results: int = 5
    ) -> List[ImageSearchResult]:
        """
        Enhanced image search that includes text analysis for PDFs.
        Uses OCR when PDFs don't have readable text.

        Args:
            search_term: Term to search for
            profile: User profile with files
            max_results: Maximum number of results to return

        Returns:
            List of ImageSearchResult with enhanced descriptions
        """
        if not profile.files:
            return []

        all_results = []

        for file in profile.files:
            if file.url.lower().endswith(".pdf"):
                try:
                    # Use enhanced search with text analysis
                    results = await self.image_processor.search_images_in_pdf_with_text_analysis(
                        file.url, search_term, max_results
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error in enhanced PDF image search {file.url}: {e}")
                    # Fallback to original method
                    try:
                        results = await self.image_processor.search_images_in_pdf(
                            file.url, search_term, max_results
                        )
                        all_results.extend(results)
                    except Exception as fallback_e:
                        print(f"Error in fallback PDF search {file.url}: {fallback_e}")
                        continue
            elif file.url.lower().endswith((".xlsx", ".xls")):
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

    async def _extract_text_from_excel_async(
        self, file_url: str, chunk_size: int = 1000
    ) -> str:
        """
        Extracts text from large Excel files asynchronously and in chunks
        """
        try:
            # Download file in chunks
            async with aiohttp.ClientSession() as session:
                async with session.get(file_url) as response:
                    response.raise_for_status()

                    # Use temporary file to handle large files
                    with tempfile.NamedTemporaryFile(
                        suffix=".xlsx", delete=False
                    ) as temp_file:
                        temp_path = temp_file.name

                        # Write in chunks to avoid memory saturation
                        async for chunk in response.content.iter_chunked(8192):
                            temp_file.write(chunk)

            # Process Excel with pandas using chunks and lazy loading
            text_parts = []
            try:
                # Read metadata first to optimize
                excel_file = pd.ExcelFile(temp_path)

                for sheet_name in excel_file.sheet_names:
                    # Read sheet by chunks for large files
                    sheet_text = f"\n--- Sheet: {sheet_name} ---\n"

                    # For very large files, read only the first N rows
                    try:
                        df = pd.read_excel(
                            temp_path, sheet_name=sheet_name, nrows=chunk_size
                        )
                        sheet_text += df.to_string(index=False, max_rows=chunk_size)

                        # If there are more rows, add a summary
                        total_rows = len(
                            pd.read_excel(temp_path, sheet_name=sheet_name, usecols=[0])
                        )
                        if total_rows > chunk_size:
                            sheet_text += f"\n... (showing first {chunk_size} of {total_rows} rows)"

                    except Exception as e:
                        sheet_text += f"Error reading sheet {sheet_name}: {e}"

                    text_parts.append(sheet_text)

                    # Yield control to event loop to avoid blocking
                    await asyncio.sleep(0)

            finally:
                excel_file.close()
                # Clean up temporary file
                os.unlink(temp_path)

            return "\n".join(text_parts)

        except Exception as e:
            return f"Error reading Excel file {file_url}: {e}"

    async def _extract_text_from_pdf_async(
        self, file_url: str, max_pages: int = 50
    ) -> str:
        """
        Extracts text from large PDF files asynchronously
        """
        try:
            # Download PDF in chunks
            async with aiohttp.ClientSession() as session:
                async with session.get(file_url) as response:
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as temp_file:
                        temp_path = temp_file.name

                        # Write in chunks
                        async for chunk in response.content.iter_chunked(8192):
                            temp_file.write(chunk)

            # Process PDF with page limit
            text_parts = []
            try:
                with open(temp_path, "rb") as file:
                    reader = PdfReader(file)
                    total_pages = len(reader.pages)

                    # Limit pages for very large files
                    pages_to_read = min(max_pages, total_pages)

                    for i in range(pages_to_read):
                        page = reader.pages[i]
                        page_text = page.extract_text() or ""
                        text_parts.append(page_text)

                        # Yield control every 10 pages
                        if i % 10 == 0:
                            await asyncio.sleep(0)

                    # Add information about unprocessed pages
                    if total_pages > max_pages:
                        text_parts.append(
                            f"\n... (showing first {max_pages} of {total_pages} pages)"
                        )

            finally:
                # Clean up temporary file
                os.unlink(temp_path)

            return "\n".join(text_parts)

        except Exception as e:
            return f"Error reading PDF file {file_url}: {e}"

    async def _extract_text_with_summary(
        self, file_url: str, max_chars: int = 50000
    ) -> str:
        """
        Extracts text with character limit and generates summary if necessary
        """
        if file_url.lower().endswith((".xlsx", ".xls")):
            text = await self._extract_text_from_excel_async(file_url)
        elif file_url.lower().endswith(".pdf"):
            text = await self._extract_text_from_pdf_async(file_url)
        else:
            return f"Unsupported format: {file_url}"

        # If text is too long, truncate and add information
        if len(text) > max_chars:
            truncated_text = text[:max_chars]
            truncated_text += f"\n\n[TEXT TRUNCATED - File too large. Showing first {max_chars} characters of {len(text)} total]"
            return truncated_text

        return text
