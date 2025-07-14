import base64
import io
import requests
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from src.core.platform.config.service import ConfigurationService
from openai import AsyncOpenAI
import asyncio
from dataclasses import dataclass
import cv2
import numpy as np
from skimage import feature, filters, measure
import matplotlib.pyplot as plt
from io import BytesIO
import re
import pytesseract
from src.core.platform.fileprocessor.processor import (
    BaseFileProcessor,
    ImageSearchResult,
)
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set module-level constants
IMAGE_CONFIDENCE_THRESHOLD = 0.3
MAX_IMAGES_TO_SHOW = 10


@dataclass
class TextExtractionResult:
    """Text extraction result with related images"""

    extracted_text: str
    related_images: List[ImageSearchResult]
    text_confidence: float
    page_number: int
    extraction_method: str  # 'pdf_text', 'ocr_embedded', 'ocr_page'


@dataclass
class TextSearchMatch:
    """Result of a text search match"""

    page_number: int
    matched_text: str
    match_context: str
    confidence: float
    start_position: int
    end_position: int


@dataclass
class SmartSearchResult:
    """Result of smart search including text analysis and related images"""

    search_term: str
    processed_keywords: List[str]
    text_matches: List[TextSearchMatch]
    related_images: List[ImageSearchResult]
    total_matches: int
    extraction_method: str


@dataclass
class PDFAnalysisResult:
    """Complete PDF analysis result"""

    has_text: bool
    total_pages: int
    text_pages: List[int]
    image_pages: List[int]
    text_extraction_results: List[TextExtractionResult]
    processing_method: str


class FileImageProcessor(BaseFileProcessor):
    """PDF file processor that extends base functionality"""

    def __init__(self, config_service: ConfigurationService):
        super().__init__(config_service)
        # Enable fast processing to reduce API calls
        self.enable_fast_processing = True

    async def analyze_pdf_content(self, pdf_url: str) -> PDFAnalysisResult:
        """
        Analyzes PDF content to determine if it has text
        and extract text from images if necessary
        """
        try:
            # Download PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_data = response.content

            # Check if PDF has text
            has_text, text_pages, total_pages = self._check_pdf_has_text(pdf_data)

            text_extraction_results = []

            if has_text:
                # If it has text, extract normal PDF text
                text_extraction_results = await self._extract_text_from_pdf(pdf_data)
                logger.info("Text extracted from PDF successfully.")
                processing_method = "pdf_text_extraction"
            else:
                # If no text, use OCR on images
                text_extraction_results = await self._extract_text_with_ocr(pdf_data)
                logger.info("Text extracted using OCR.")
                processing_method = "ocr_extraction"

            # Identify pages with images
            image_pages = await self._identify_image_pages(pdf_data)

            return PDFAnalysisResult(
                has_text=has_text,
                total_pages=total_pages,
                text_pages=text_pages,
                image_pages=image_pages,
                text_extraction_results=text_extraction_results,
                processing_method=processing_method,
            )

        except Exception as e:
            raise Exception(f"Error analyzing PDF content: {str(e)}")

    def _check_pdf_has_text(self, pdf_data: bytes) -> Tuple[bool, List[int], int]:
        """
        Checks if the PDF has readable text
        """
        try:
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            text_pages = []
            total_pages = len(pdf_document)

            for page_num in range(total_pages):
                page = pdf_document[page_num]
                text = page.get_text().strip()

                # Check if there's significant text (more than 10 characters)
                if len(text) > 10:
                    text_pages.append(page_num + 1)

            pdf_document.close()

            has_text = len(text_pages) > 0
            return has_text, text_pages, total_pages

        except Exception as e:
            logger.error(f"Error checking PDF text: {e}")
            return False, [], 0

    async def _extract_text_from_pdf(
        self, pdf_data: bytes
    ) -> List[TextExtractionResult]:
        """
        Extracts text from PDF using traditional methods
        """
        try:
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            results = []

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text().strip()

                if len(text) > 10:  # Only process pages with significant text
                    # Find related images on the same page
                    related_images = await self._find_related_images_in_page(
                        pdf_data, page_num + 1, text
                    )

                    result = TextExtractionResult(
                        extracted_text=text,
                        related_images=related_images,
                        text_confidence=1.0,  # High confidence for native PDF text
                        page_number=page_num + 1,
                        extraction_method="pdf_text",
                    )
                    results.append(result)

            pdf_document.close()
            return results

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return []

    async def _extract_text_with_ocr(
        self, pdf_data: bytes
    ) -> List[TextExtractionResult]:
        """
        Extracts text using OCR when PDF has no readable text
        """
        try:
            results = []

            # First try with embedded images
            embedded_results = await self._extract_text_from_embedded_images(pdf_data)
            results.extend(embedded_results)

            # If not enough text from embedded images, convert full pages
            if (
                len(embedded_results) == 0
                or sum(len(r.extracted_text) for r in embedded_results) < 100
            ):
                page_results = await self._extract_text_from_full_pages(pdf_data)
                results.extend(page_results)

            return results

        except Exception as e:
            logger.error(f"Error extracting text with OCR: {e}")
            return []

    async def _extract_text_from_embedded_images(
        self, pdf_data: bytes
    ) -> List[TextExtractionResult]:
        """
        Extracts text from embedded images in the PDF using OCR
        """
        try:
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            results = []

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)

                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))

                            # Apply OCR
                            extracted_text = self._extract_text_with_tesseract(
                                pil_image
                            )

                            if (
                                len(extracted_text.strip()) > 10
                            ):  # Only if there's significant text
                                # Convert image to base64 for result
                                buffered = io.BytesIO()
                                pil_image.save(buffered, format="PNG")
                                img_base64 = base64.b64encode(
                                    buffered.getvalue()
                                ).decode()

                                # Find related images
                                related_images = (
                                    await self._find_related_images_in_page(
                                        pdf_data, page_num + 1, extracted_text
                                    )
                                )

                                # Add current image as related
                                current_image = ImageSearchResult(
                                    page_number=page_num + 1,
                                    sheet_name=None,
                                    image_base64=img_base64,
                                    description=f"Image with extracted text: {extracted_text[:100]}...",
                                    confidence=0.9,
                                )
                                related_images.insert(0, current_image)

                                result = TextExtractionResult(
                                    extracted_text=extracted_text,
                                    related_images=related_images,
                                    text_confidence=self._calculate_ocr_confidence(
                                        extracted_text
                                    ),
                                    page_number=page_num + 1,
                                    extraction_method="ocr_embedded",
                                )
                                results.append(result)

                        pix = None

                    except Exception as e:
                        logger.error(
                            f"Error processing embedded image {img_index} on page {page_num + 1}: {e}"
                        )
                        continue

            pdf_document.close()
            return results

        except Exception as e:
            logger.error(f"Error extracting text from embedded images: {e}")
            return []

    async def _extract_text_from_full_pages(
        self, pdf_data: bytes
    ) -> List[TextExtractionResult]:
        """
        Extracts text by converting full PDF pages to images and applying OCR
        """
        try:
            # Convert pages to images
            images = convert_from_bytes(
                pdf_data, dpi=300
            )  # High resolution for better OCR
            results = []

            for page_num, image in enumerate(images):
                try:
                    # Apply OCR to the full page
                    extracted_text = self._extract_text_with_tesseract(image)

                    if (
                        len(extracted_text.strip()) > 20
                    ):  # Only if there's significant text
                        # Convert image to base64
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()

                        # Find related images on the page
                        related_images = await self._find_related_images_in_page(
                            pdf_data, page_num + 1, extracted_text
                        )

                        # Add the full page as a related image
                        page_image = ImageSearchResult(
                            page_number=page_num + 1,
                            sheet_name=None,
                            image_base64=img_base64,
                            description=f"Full page with text extracted via OCR",
                            confidence=0.8,
                        )
                        related_images.insert(0, page_image)

                        result = TextExtractionResult(
                            extracted_text=extracted_text,
                            related_images=related_images,
                            text_confidence=self._calculate_ocr_confidence(
                                extracted_text
                            ),
                            page_number=page_num + 1,
                            extraction_method="ocr_page",
                        )
                        results.append(result)

                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Error extracting text from full pages: {e}")
            return []

    def _extract_text_with_tesseract(self, image: Image.Image) -> str:
        """
        Extracts text from an image using Tesseract OCR
        """
        try:
            # Preprocess image to improve OCR
            processed_image = self._preprocess_image_for_ocr(image)

            # Tesseract configuration for Spanish and English
            custom_config = r"--oem 3 --psm 6 -l spa+eng"

            # Extract text
            text = pytesseract.image_to_string(processed_image, config=custom_config)

            # Clean text
            cleaned_text = self._clean_extracted_text(text)

            return cleaned_text

        except Exception as e:
            logger.error(f"Error with Tesseract OCR: {e}")
            # Fallback without specific configuration
            try:
                text = pytesseract.image_to_string(image)
                return self._clean_extracted_text(text)
            except:
                return ""

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocesses image to improve OCR accuracy
        """
        try:
            # Convert to grayscale if necessary
            if image.mode != "L":
                image = image.convert("L")

            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)

            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)

            # Resize if too small (minimum 300px width)
            width, height = image.size
            if width < 300:
                ratio = 300 / width
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.LANCZOS)

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image

    def _clean_extracted_text(self, text: str) -> str:
        """
        Cleans the OCR extracted text
        """
        try:
            # Remove strange characters and excessive spaces
            cleaned = re.sub(
                r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=]", " ", text
            )

            # Remove multiple spaces
            cleaned = re.sub(r"\s+", " ", cleaned)

            # Remove multiple empty lines
            lines = cleaned.split("\n")
            cleaned_lines = [line.strip() for line in lines if line.strip()]

            return "\n".join(cleaned_lines).strip()

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text.strip()

    def _calculate_ocr_confidence(self, text: str) -> float:
        """
        Calculates OCR confidence based on text characteristics
        """
        try:
            if not text.strip():
                return 0.0

            confidence = 0.5  # Base confidence

            # Bonus for reasonable length
            if 50 <= len(text) <= 5000:
                confidence += 0.2
            elif len(text) > 20:
                confidence += 0.1

            # Bonus for presence of complete words
            words = text.split()
            if len(words) > 5:
                confidence += 0.1

            # Bonus for common characters in Spanish/English
            common_chars = sum(
                1 for c in text.lower() if c in "abcdefghijklmnopqrstuvwxyzáéíóúñü"
            )
            char_ratio = common_chars / len(text) if text else 0
            confidence += char_ratio * 0.2

            # Penalty for many weird characters
            weird_chars = sum(
                1 for c in text if ord(c) > 127 and c not in "áéíóúñüÁÉÍÓÚÑÜ"
            )
            if weird_chars > len(text) * 0.3:
                confidence -= 0.2

            return min(max(confidence, 0.0), 1.0)

        except Exception:
            return 0.5

    async def _find_related_images_in_page(
        self, pdf_data: bytes, page_number: int, text: str
    ) -> List[ImageSearchResult]:
        """
        Finds images related to text on the same page
        """
        try:
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            page = pdf_document[page_number - 1]  # page_number is 1-indexed
            related_images = []

            # Extract all images from the page
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)

                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))

                        # Convert to base64
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()

                        # Analyze image relevance to text
                        if getattr(self, 'enable_fast_processing', True):
                            # Use optimized method (no API calls)
                            relevance_score = await self._calculate_image_text_relevance_optimized(
                                img_base64, text
                            )
                        else:
                            # Use full method (with API calls)
                            relevance_score = await self._calculate_image_text_relevance(
                                img_base64, text
                            )

                        if (
                            relevance_score > 0.3
                        ):  # Significantly reduced threshold for better recall
                            description = (
                                await self._generate_image_description_with_context(
                                    img_base64, text
                                )
                            )

                            image_result = ImageSearchResult(
                                page_number=page_number,
                                sheet_name=None,
                                image_base64=img_base64,
                                description=description,
                                confidence=relevance_score,
                            )
                            related_images.append(image_result)

                    pix = None

                except Exception as e:
                    logger.error(
                        f"Error processing image {img_index} on page {page_number}: {e}"
                    )
                    continue

            pdf_document.close()

            # Sort by relevance
            related_images.sort(key=lambda x: x.confidence, reverse=True)

            return related_images[:5]  # Maximum 5 related images

        except Exception as e:
            logger.error(f"Error finding related images: {e}")
            return []

    async def _calculate_image_text_relevance(
        self, image_base64: str, text: str
    ) -> float:
        """
        Calculates the relevance of an image relative to extracted text
        """
        try:
            # Technical analysis of the image
            technical_analysis = self._analyze_image_with_python_tools(image_base64)

            base_relevance = 0.4  # Base relevance

            # Keywords from text for contextual analysis
            text_keywords = self._extract_keywords_from_text(text)

            # Bonus for image quality
            quality_score = technical_analysis.get("quality_score", 0.0)
            if quality_score > 0.7:
                base_relevance += 0.2

            # Bonus for detected objects
            objects_detected = technical_analysis.get("objects_detected", 0)
            if objects_detected > 0:
                base_relevance += 0.2

            # Bonus if there's text in the image
            if technical_analysis.get("text_detected", False):
                base_relevance += 0.3

            # Basic semantic analysis with AI
            semantic_relevance = await self._analyze_semantic_relevance(
                technical_analysis, text_keywords
            )

            final_relevance = (base_relevance + semantic_relevance) / 2
            return min(max(final_relevance, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating image-text relevance: {e}")
            return 0.4

    async def _calculate_image_text_relevance_optimized(
        self, image_base64: str, text: str
    ) -> float:
        """
        Optimized calculation of image relevance relative to extracted text (no API calls)
        """
        try:
            # Extract keywords from text
            text_keywords = self._extract_keywords_from_text(text)

            # Analyze image with Python tools only
            technical_analysis = self._analyze_image_with_python_tools(image_base64)

            # Use optimized relevance calculation
            relevance_score = self._calculate_relevance_score_optimized(
                technical_analysis, " ".join(text_keywords[:5])  # Use top 5 keywords
            )

            return relevance_score

        except Exception as e:
            logger.error(f"Error in optimized relevance calculation: {e}")
            return 0.5

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """
        Extracts keywords from text
        """
        try:
            # Common words to filter out
            stopwords = {
                "el",
                "la",
                "de",
                "que",
                "y",
                "en",
                "un",
                "es",
                "se",
                "no",
                "te",
                "lo",
                "le",
                "da",
                "su",
                "por",
                "son",
                "con",
                "para",
                "al",
                "del",
                "las",
                "los",
                "una",
                "pero",
                "sus",
                "como",
                "si",
                "ya",
                "me",
                "mi",
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
            }

            # Extract words
            words = re.findall(r"\b[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]{3,}\b", text.lower())

            # Filter stopwords and very common words
            keywords = [
                word for word in words if word not in stopwords and len(word) > 3
            ]

            # Count frequencies and take the most common
            from collections import Counter

            word_counts = Counter(keywords)

            # Return the 10 most frequent words
            return [word for word, count in word_counts.most_common(10)]

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    async def _analyze_semantic_relevance(
        self, technical_analysis: Dict[str, Any], keywords: List[str]
    ) -> float:
        """
        Analyzes semantic relevance using AI
        """
        try:
            keywords_str = ", ".join(keywords[:5])  # Only the top 5

            prompt = f"""
            Analyze the relevance of an image based on its technical analysis and text keywords:

            TECHNICAL ANALYSIS:
            - Quality: {technical_analysis.get('quality_score', 0):.2f}/1.0
            - Objects detected: {technical_analysis.get('objects_detected', 0)}
            - Text detected: {'Yes' if technical_analysis.get('text_detected', False) else 'No'}
            - Dominant colors: {len(technical_analysis.get('dominant_colors', []))}

            TEXT KEYWORDS: {keywords_str}

            Based on this information, assign a relevance score from 0.0 to 1.0:
            - 0.0-0.3: Low relevance (decorative or unrelated image)
            - 0.4-0.6: Medium relevance (possible indirect relationship)
            - 0.7-1.0: High relevance (clearly related to content)

            Respond with only a number between 0.0 and 1.0:
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )

            content = response.choices[0].message.content.strip()

            # Extract number from response
            try:
                relevance = float(re.findall(r"0\.\d+|1\.0", content)[0])
                return min(max(relevance, 0.0), 1.0)
            except:
                return 0.5  # Default value

        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return 0.5

    async def _generate_image_description_with_context(
        self, image_base64: str, context_text: str
    ) -> str:
        """
        Generates image description considering text context
        """
        try:
            # Technical analysis
            technical_analysis = self._analyze_image_with_python_tools(image_base64)

            # Extract keywords from context
            keywords = self._extract_keywords_from_text(context_text)
            keywords_str = ", ".join(keywords[:5])

            # Create enhanced description
            base_description = self._create_detailed_description(technical_analysis)

            prompt = f"""
            Improve this technical image description considering the text context:

            TECHNICAL DESCRIPTION:
            {base_description}

            TEXT CONTEXT (keywords): {keywords_str}

            RELATED TEXT (first 200 chars): {context_text[:200]}...

            Instructions:
            1. Create a more natural and contextual description
            2. Suggest how the image relates to the text
            3. Keep relevant technical information
            4. Maximum 150 words
            5. Respond in English

            Enhanced description:
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )

            enhanced_description = response.choices[0].message.content.strip()
            return enhanced_description

        except Exception as e:
            logger.error(f"Error generating enhanced description: {e}")
            return f"Image related to text. Technical analysis available."

    async def _identify_image_pages(self, pdf_data: bytes) -> List[int]:
        """
        Identifies which PDF pages contain images
        """
        try:
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            image_pages = []

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()

                if len(image_list) > 0:
                    image_pages.append(page_num + 1)

            pdf_document.close()
            return image_pages

        except Exception as e:
            logger.error(f"Error identifying image pages: {e}")
            return []

    async def extract_text_and_images(
        self,
        pdf_url: str,
        search_term: Optional[str] = None,
        has_image_processor: str = None,
    ) -> Dict[str, Any]:
        """
        Main method to extract text and images from a PDF.
        If the PDF has no text, uses OCR. Also identifies related images.

        Args:
            pdf_url: URL of the PDF to process
            search_term: Optional search term to filter images
            has_image_processor: Control flag for image processing ("activate" to enable page screenshots)

        Returns:
            Dictionary with extracted text, related images and metadata
        """
        try:
            # Complete PDF analysis
            analysis_result = await self.analyze_pdf_content(pdf_url)

            # Process results
            all_extracted_text = []
            all_related_images = []

            for text_result in analysis_result.text_extraction_results:
                all_extracted_text.append(
                    {
                        "page": text_result.page_number,
                        "text": text_result.extracted_text,
                        "confidence": text_result.text_confidence,
                        "method": text_result.extraction_method,
                        "related_images_count": len(text_result.related_images),
                    }
                )

                # Add related images
                for img in text_result.related_images:
                    all_related_images.append(
                        {
                            "page": img.page_number,
                            "image_base64": img.image_base64,
                            "description": img.description,
                            "confidence": img.confidence,
                            "related_to_text": text_result.extracted_text[:100] + "...",
                        }
                    )

            # If search term provided, filter relevant images
            filtered_images = all_related_images
            if search_term:
                filtered_images = await self._filter_images_by_search_term(
                    all_related_images, search_term
                )

                # If no images found but search term exists, try forced extraction
                if len(filtered_images) == 0:
                    logger.info(
                        f"No images found for '{search_term}', trying forced extraction..."
                    )
                    # Download PDF data for fallback methods
                    try:
                        response = requests.get(pdf_url)
                        response.raise_for_status()
                        pdf_data = response.content

                        forced_images = await self._force_search_images_for_term(
                            pdf_data, search_term, all_extracted_text
                        )
                        if forced_images:
                            filtered_images = forced_images
                            logger.info(
                                f"Forced extraction found {len(forced_images)} images"
                            )
                        else:
                            logger.info(
                                "Forced extraction also returned no images, trying direct search..."
                            )
                            direct_images = await self._search_images_directly_in_pdf(
                                pdf_data, search_term
                            )
                            if direct_images:
                                filtered_images = direct_images
                                logger.info(
                                    f"Direct search found {len(direct_images)} images"
                                )
                            else:
                                logger.info(
                                    "All extraction methods failed to find images"
                                )

                                # Last resort: capture page screenshots if has_image_processor is activate
                                if has_image_processor == "activate":
                                    logger.info(
                                        f"Trying page screenshot capture for '{search_term}'..."
                                    )
                                    page_screenshots = (
                                        await self._capture_page_screenshots_for_term(
                                            pdf_data,
                                            search_term,
                                            all_extracted_text,
                                            has_image_processor,
                                        )
                                    )
                                    if page_screenshots:
                                        filtered_images = page_screenshots
                                        logger.info(
                                            f"Page screenshot capture found {len(page_screenshots)} page images"
                                        )
                                    else:
                                        logger.info(
                                            "Page screenshot capture also returned no images"
                                        )
                                else:
                                    logger.info(
                                        "Page screenshot capture is disabled (has_image_processor != 'activate')"
                                    )
                    except Exception as e:
                        logger.error(f"Error in fallback methods: {e}")

            # Create summary of extracted text
            full_text = "\n\n".join([item["text"] for item in all_extracted_text])
            text_summary = await self._create_text_summary(full_text)

            result = {
                "analysis": {
                    "has_original_text": analysis_result.has_text,
                    "total_pages": analysis_result.total_pages,
                    "text_pages": analysis_result.text_pages,
                    "image_pages": analysis_result.image_pages,
                    "processing_method": analysis_result.processing_method,
                },
                "extracted_text": {
                    "full_text": full_text,
                    "summary": text_summary,
                    "pages_detail": all_extracted_text,
                    "total_characters": len(full_text),
                    "avg_confidence": (
                        sum(item["confidence"] for item in all_extracted_text)
                        / len(all_extracted_text)
                        if all_extracted_text
                        else 0
                    ),
                },
                "related_images": {
                    "total_found": len(all_related_images),
                    "filtered_count": len(filtered_images),
                    "images": filtered_images[:10],  # Maximum 10 images
                    "search_term": search_term,
                },
                "recommendations": self._generate_processing_recommendations(
                    analysis_result
                ),
            }

            return result

        except Exception as e:
            return {
                "error": f"Error processing PDF: {str(e)}",
                "analysis": None,
                "extracted_text": None,
                "related_images": None,
                "recommendations": ["Manual review required"],
            }

    async def _filter_images_by_search_term(
        self, images: List[Dict[str, Any]], search_term: str
    ) -> List[Dict[str, Any]]:
        """
        Filters images based on a search term with improved matching for products
        """
        try:
            filtered = []
            search_term_lower = search_term.lower()
            search_words = search_term_lower.split()

            logger.info(
                f"DEBUG: Filtering {len(images)} images for search term '{search_term}'"
            )

            for img_idx, img in enumerate(images):
                relevance_score = 0.0
                match_reasons = []

                # Check in description (direct text match)
                description_lower = img["description"].lower()
                if search_term_lower in description_lower:
                    relevance_score += 0.4
                    match_reasons.append("description_exact")
                elif any(
                    word in description_lower for word in search_words if len(word) > 2
                ):
                    relevance_score += 0.3
                    match_reasons.append("description_partial")

                # Check in related text (context from OCR or PDF text)
                related_text_lower = img["related_to_text"].lower()
                if search_term_lower in related_text_lower:
                    relevance_score += 0.5  # Higher weight for direct context
                    match_reasons.append("context_exact")
                elif any(
                    word in related_text_lower for word in search_words if len(word) > 2
                ):
                    relevance_score += 0.35
                    match_reasons.append("context_partial")

                # Check for similar/related words (plurals, variations)
                if self._contains_similar_words(search_term_lower, description_lower):
                    relevance_score += 0.25
                    match_reasons.append("description_similar")

                if self._contains_similar_words(search_term_lower, related_text_lower):
                    relevance_score += 0.30
                    match_reasons.append("context_similar")

                # Additional semantic analysis for stronger confidence
                if relevance_score > 0.1:  # Lowered threshold for semantic analysis
                    semantic_score = await self._calculate_search_relevance(
                        img["description"], search_term
                    )
                    relevance_score += semantic_score * 0.3  # Weight semantic analysis

                    if semantic_score > 0.7:
                        match_reasons.append("semantic_strong")
                    elif semantic_score > 0.5:
                        match_reasons.append("semantic_moderate")

                # Boost confidence based on match quality
                if relevance_score > 0.0:
                    # Calculate final confidence
                    base_confidence = img.get("confidence", 0.0)

                    # Boost confidence based on relevance
                    confidence_boost = min(relevance_score, 0.4)  # Increased boost
                    final_confidence = min(base_confidence + confidence_boost, 1.0)

                    # Debug output for each image
                    logger.info(
                        f"  Image {img_idx + 1} (Page {img.get('page', '?')}): Base conf: {base_confidence:.3f}, "
                        f"Relevance: {relevance_score:.3f}, Final: {final_confidence:.3f}, "
                        f"Reasons: {match_reasons}"
                    )

                    # Very permissive filtering to maximize recall for product catalogs
                    if (
                        final_confidence > 0.3 and relevance_score > 0.05
                    ):  # Much lower thresholds for better recall
                        img_copy = img.copy()
                        img_copy["confidence"] = final_confidence
                        img_copy["search_relevance"] = relevance_score
                        img_copy["match_reasons"] = match_reasons

                        # Update description to include match information
                        match_info = f" (matched: {', '.join(match_reasons[:2])})"
                        img_copy["description"] = img["description"] + match_info

                        filtered.append(img_copy)
                        logger.info(f"    -> INCLUDED in results")
                    else:
                        logger.info(
                            f"    -> EXCLUDED (conf: {final_confidence:.3f} <= 0.3 or relevance: {relevance_score:.3f} <= 0.05)"
                        )
                else:
                    logger.info(
                        f"  Image {img_idx + 1}: No relevance found (score: {relevance_score:.3f})"
                    )

            # Sort by search relevance first, then confidence
            filtered.sort(
                key=lambda x: (x.get("search_relevance", 0), x["confidence"]),
                reverse=True,
            )

            # Debug information
            logger.info(
                f"RESULT: Filtered {len(filtered)} images from {len(images)} total for search term '{search_term}'"
            )

            if len(filtered) == 0:
                logger.info("No images passed the filtering criteria. Debug info:")
                for i, img in enumerate(images[:3]):  # Show first 3 for debugging
                    desc = img.get("description", "")
                    related = img.get("related_to_text", "")
                    conf = img.get("confidence", 0)
                    logger.info(f"  Image {i+1}: conf={conf:.3f}")
                    logger.info(f"    Description: {desc[:100]}...")
                    logger.info(f"    Related text: {related[:100]}...")
                    logger.info(
                        f"    Contains '{search_term}': desc={search_term.lower() in desc.lower()}, text={search_term.lower() in related.lower()}"
                    )
            else:
                for i, img in enumerate(filtered[:3]):  # Show top 3
                    logger.info(
                        f"  {i+1}. Page {img.get('page', '?')}, Confidence: {img['confidence']:.3f}, "
                        f"Relevance: {img.get('search_relevance', 0):.3f}, Reasons: {img.get('match_reasons', [])}"
                    )

            return filtered

        except Exception as e:
            logger.error(f"Error filtering images: {e}")
            return images

    def _contains_similar_words(self, search_term: str, text: str) -> bool:
        """Check if text contains words similar to search term"""
        try:
            search_words = search_term.split()

            for search_word in search_words:
                if len(search_word) > 2 and self._is_similar_word(search_word, text):
                    return True

            return False

        except Exception:
            return False

    def _is_similar_word(self, search_word: str, text: str) -> bool:
        """Check if search word is similar to words in text (handles plurals, etc.)"""
        try:
            search_word = search_word.strip()
            if len(search_word) < 3:
                return False

            # Split text into words
            text_words = re.findall(r"\b\w+\b", text.lower())

            for word in text_words:
                # Exact match
                if search_word == word:
                    return True

                # Check for plural/singular variations (more comprehensive)
                if len(search_word) >= 3 and len(word) >= 3:
                    # Remove common Spanish endings
                    search_root = re.sub(r"(s|es|as|a|o|os)$", "", search_word)
                    word_root = re.sub(r"(s|es|as|a|o|os)$", "", word)

                    if search_root == word_root and len(search_root) >= 3:
                        return True

                # Check for partial matches in longer words (more permissive)
                if len(search_word) >= 3 and search_word in word:
                    return True

                # Reverse check - check if word is contained in search term
                if len(word) >= 3 and word in search_word:
                    return True

            return False

        except Exception:
            return False

    async def _calculate_search_relevance(
        self, description: str, search_term: str
    ) -> float:
        """
        Calculates relevance of a description to a search term
        """
        try:
            prompt = f"""
            Calculate semantic relevance between an image description and a search term:

            DESCRIPTION: {description}
            SEARCH TERM: {search_term}

            Assign a score from 0.0 to 1.0 where:
            - 0.0-0.2: No relation
            - 0.3-0.5: Indirect relation
            - 0.6-0.8: Direct relation
            - 0.9-1.0: Exact match

            Respond with only a number:
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )

            content = response.choices[0].message.content.strip()

            try:
                relevance = float(re.findall(r"0\.\d+|1\.0", content)[0])
                return min(max(relevance, 0.0), 1.0)
            except:
                return 0.3  # Default value

        except Exception as e:
            logger.error(f"Error calculating search relevance: {e}")
            return 0.3

    async def _create_text_summary(self, full_text: str) -> str:
        """
        Creates a summary of extracted text
        """
        try:
            if len(full_text) < 100:
                return full_text

            prompt = f"""
            Create a concise summary of the following text extracted from a PDF:

            TEXT:
            {full_text[:2000]}...

            Instructions:
            1. Summary in 3-5 sentences
            2. Capture main points
            3. Maintain original context
            4. Respond in English
            5. Maximum 200 words

            Summary:
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return full_text[:300] + "..." if len(full_text) > 300 else full_text

    def _generate_processing_recommendations(
        self, analysis_result: PDFAnalysisResult
    ) -> List[str]:
        """
        Generates recommendations based on PDF analysis
        """
        recommendations = []

        try:
            if not analysis_result.has_text:
                recommendations.append(
                    "PDF does not contain readable text. OCR was used to extract text from images."
                )

                if len(analysis_result.text_extraction_results) == 0:
                    recommendations.append(
                        "No significant text could be extracted. Consider verifying image quality."
                    )
                else:
                    avg_confidence = sum(
                        r.text_confidence
                        for r in analysis_result.text_extraction_results
                    ) / len(analysis_result.text_extraction_results)

                    if avg_confidence < 0.6:
                        recommendations.append(
                            "OCR confidence is low. Extracted text may contain errors."
                        )
                    else:
                        recommendations.append(
                            "OCR completed with good confidence. Extracted text is reliable."
                        )
            else:
                recommendations.append(
                    "PDF contains native text. Text extraction completed successfully."
                )

            if len(analysis_result.image_pages) > 0:
                recommendations.append(
                    f"Images found on {len(analysis_result.image_pages)} pages. "
                    f"Images have been analyzed and related to text."
                )

            if analysis_result.processing_method == "ocr_extraction":
                recommendations.append(
                    "To improve OCR results, consider using higher resolution PDFs or sharper images."
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Processing completed. Manual review recommended."]

    # Convenience method for compatibility with existing code
    async def search_images_in_pdf_with_text_analysis(
        self,
        pdf_url: str,
        search_term: str,
        max_results: int = 5,
        has_image_processor: str = None,
    ) -> List[ImageSearchResult]:
        """
        Enhanced search that considers both extracted text and images with spatial proximity
        """
        try:
            logger.info(f"Starting enhanced PDF search for term: '{search_term}'")

            # Download PDF to get data for position analysis
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_data = response.content

            # Extract text and image positions for proximity analysis
            logger.info("Extracting text and image positions...")
            positions_data = await self._extract_text_and_image_positions_pdf(pdf_data)

            # Use the main method to extract text and images
            logger.info("Analyzing PDF content and extracting images...")
            result = await self.extract_text_and_images(
                pdf_url, search_term, has_image_processor
            )

            if not result.get("related_images", {}).get("images"):
                logger.info(
                    "No images found in initial analysis. Trying direct image search..."
                )
                # Fallback: try to find images directly from PDF
                direct_images = await self._search_images_directly_in_pdf(
                    pdf_data, search_term
                )
                if direct_images:
                    # Convert to expected format
                    return direct_images[:max_results]

                # Last resort: force search for images on pages with matching text
                logger.info("Direct search also failed. Trying force search...")
                # Create mock extracted text data for force search
                mock_extracted_text = [
                    {"page": i + 1, "text": ""} for i in range(10)
                ]  # Assume up to 10 pages
                force_images = await self._force_search_images_for_term(
                    pdf_data, search_term, mock_extracted_text
                )
                if force_images:
                    return force_images[:max_results]

                # Even more aggressive: extract ALL images
                logger.info("Force search also failed. Trying aggressive extraction...")
                aggressive_images = await self._extract_all_images_aggressive(
                    pdf_data, search_term
                )
                if aggressive_images:
                    # Convert to expected format and return
                    converted_images = []
                    for img_data in aggressive_images[:max_results]:
                        img_result = ImageSearchResult(
                            page_number=img_data["page"],
                            image_base64=img_data["image_base64"],
                            description=img_data["description"],
                            confidence=img_data["confidence"],
                        )
                        converted_images.append(img_result)
                    return converted_images

                # Final fallback: capture page screenshots if enabled
                if has_image_processor == "activate":
                    logger.info(
                        "Aggressive extraction also failed. Trying page screenshot capture..."
                    )
                    page_screenshots = await self._capture_page_screenshots_for_term(
                        pdf_data, search_term, mock_extracted_text, has_image_processor
                    )
                    if page_screenshots:
                        # Convert to expected format
                        screenshot_results = []
                        for img_data in page_screenshots[:max_results]:
                            screenshot_result = ImageSearchResult(
                                page_number=img_data["page"],
                                image_base64=img_data["image_base64"],
                                description=img_data["description"],
                                confidence=img_data["confidence"],
                            )
                            screenshot_results.append(screenshot_result)
                        return screenshot_results

            # Enhance confidence scores with proximity analysis
            enhanced_images = []
            images_found = result.get("related_images", {}).get("images", [])
            logger.info(f"Found {len(images_found)} images to analyze for proximity...")

            for img_data in images_found:
                # Calculate proximity bonus based on spatial closeness to search term
                proximity_bonus = self._calculate_pdf_text_proximity(
                    search_term, img_data, positions_data
                )

                logger.info(
                    f"Image on page {img_data.get('page', '?')} - Original confidence: {img_data.get('confidence', 0):.3f}, Proximity bonus: {proximity_bonus:.3f}"
                )

                # Adjust confidence with proximity bonus
                enhanced_confidence = min(img_data["confidence"] + proximity_bonus, 1.0)

                # Update description to include proximity information
                proximity_desc = ""
                if proximity_bonus > 0.2:
                    proximity_desc = " (very close to matching text)"
                elif proximity_bonus > 0.1:
                    proximity_desc = " (near matching text)"
                elif proximity_bonus > 0.05:
                    proximity_desc = " (close to relevant content)"

                enhanced_images.append(
                    {
                        **img_data,
                        "confidence": enhanced_confidence,
                        "description": img_data["description"] + proximity_desc,
                        "proximity_bonus": proximity_bonus,
                    }
                )

            # Sort by enhanced confidence
            enhanced_images.sort(key=lambda x: x["confidence"], reverse=True)

            # Convert to expected format
            image_results = []
            for i, img_data in enumerate(enhanced_images[:max_results]):
                logger.info(
                    f"Result {i+1}: Page {img_data['page']}, Final confidence: {img_data['confidence']:.3f}"
                )

                image_result = ImageSearchResult(
                    page_number=img_data["page"],
                    image_base64=img_data["image_base64"],
                    description=img_data["description"],
                    confidence=img_data["confidence"],
                )
                image_results.append(image_result)

            logger.info(f"Returning {len(image_results)} enhanced results")
            return image_results

        except Exception as e:
            logger.error(f"Error in enhanced image search: {e}")
            # Fallback to original method
            return await self.search_images_in_pdf(pdf_url, search_term, max_results)

    async def smart_search_with_text_analysis(
        self,
        file_url: str,
        search_phrase: str,
        file_type: str = "pdf",
        max_results: int = 10,
    ) -> SmartSearchResult:
        """
        Búsqueda inteligente que:
        1. Procesa el término de búsqueda (puede ser una frase) usando regex
        2. Extrae todo el texto del archivo (PDF o XLSX)
        3. Busca coincidencias de palabras clave en el texto
        4. Devuelve las imágenes relacionadas basándose en las coincidencias

        Args:
            file_url: URL del archivo (PDF o XLSX)
            search_phrase: Término o frase de búsqueda
            file_type: Tipo de archivo ("pdf" o "xlsx")
            max_results: Número máximo de imágenes a devolver

        Returns:
            SmartSearchResult con análisis de texto y imágenes relacionadas
        """
        try:
            logger.info(
                f"🔍 Iniciando búsqueda inteligente para: '{search_phrase}' en archivo {file_type.upper()}"
            )

            # 1. Procesar el término de búsqueda con regex para extraer palabras clave
            keywords = self._extract_keywords_from_phrase(search_phrase)
            logger.info(f"📝 Palabras clave extraídas: {keywords}")

            # 2. Extraer texto del archivo según el tipo
            if file_type.lower() == "pdf":
                extracted_text, extraction_method = (
                    await self._extract_all_text_from_pdf(file_url)
                )
            elif file_type.lower() == "xlsx":
                extracted_text, extraction_method = (
                    await self._extract_all_text_from_xlsx(file_url)
                )
            else:
                raise ValueError(f"Tipo de archivo no soportado: {file_type}")

            logger.info(
                f"📄 Texto extraído: {len(extracted_text)} caracteres usando método: {extraction_method}"
            )

            # 3. Buscar coincidencias en el texto usando regex
            text_matches = self._find_text_matches_with_regex(
                extracted_text, keywords, search_phrase
            )
            logger.info(f"🎯 Encontradas {len(text_matches)} coincidencias de texto")

            # 4. Buscar imágenes relacionadas basándose en las coincidencias
            related_images = await self._find_images_for_text_matches(
                file_url, text_matches, search_phrase, file_type, max_results
            )
            logger.info(f"🖼️ Encontradas {len(related_images)} imágenes relacionadas")

            # 5. Construir resultado final
            result = SmartSearchResult(
                search_term=search_phrase,
                processed_keywords=keywords,
                text_matches=text_matches,
                related_images=related_images,
                total_matches=len(text_matches),
                extraction_method=extraction_method,
            )

            return result

        except Exception as e:
            logger.error(f"❌ Error en búsqueda inteligente: {str(e)}")
            # Retornar resultado vacío en caso de error
            return SmartSearchResult(
                search_term=search_phrase,
                processed_keywords=[],
                text_matches=[],
                related_images=[],
                total_matches=0,
                extraction_method="error",
            )

    def _extract_keywords_from_phrase(self, search_phrase: str) -> List[str]:
        """
        Extract keywords from a phrase using regex
        """
        try:
            # Clean and normalize the phrase
            cleaned_phrase = search_phrase.lower().strip()

            # Use regex to extract words (letters, numbers and some special characters only)
            # Exclude very short words and common articles/prepositions
            stopwords = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "up",
                "about",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "among",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "should",
                "could",
                "can",
                "may",
                "might",
                "must",
                "shall",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
                "my",
                "your",
                "his",
                "its",
                "our",
                "their",
                "if",
                "so",
                "as",
                "no",
                "not",
                "more",
                "most",
                "less",
                "like",
                "than",
            }

            # Extract words using regex (minimum 2 characters)
            words = re.findall(r"\b[a-z0-9]{2,}\b", cleaned_phrase, re.IGNORECASE)

            # Filter stopwords and duplicates
            keywords = []
            seen = set()

            for word in words:
                word_lower = word.lower()
                if word_lower not in stopwords and word_lower not in seen:
                    keywords.append(word)
                    seen.add(word_lower)

            # If no valid words found, use the complete phrase
            if not keywords:
                keywords = [search_phrase.strip()]

            return keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return [search_phrase.strip()]

    async def _extract_all_text_from_pdf(self, pdf_url: str) -> Tuple[str, str]:
        """
        Extrae todo el texto de un PDF usando múltiples métodos
        """
        try:
            # Descargar PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_data = response.content

            # Primero intentar extracción nativa de PDF
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            native_text = ""

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text().strip()
                if page_text:
                    native_text += f"\n[PÁGINA {page_num + 1}]\n{page_text}\n"

            pdf_document.close()

            # Si hay texto nativo suficiente, usarlo
            if len(native_text.strip()) > 50:
                return native_text, "pdf_native_text"

            # Si no hay texto nativo, usar OCR
            logger.info("📖 Texto nativo insuficiente, usando OCR...")
            ocr_text = await self._extract_text_with_ocr_comprehensive(pdf_data)

            if ocr_text:
                return ocr_text, "ocr_extraction"
            else:
                return native_text, "pdf_native_text_limited"

        except Exception as e:
            logger.error(f"Error extrayendo texto de PDF: {e}")
            return "", "error"

    async def _extract_all_text_from_xlsx(self, xlsx_url: str) -> Tuple[str, str]:
        """
        Extrae todo el texto de un archivo XLSX
        """
        try:
            try:
                import pandas as pd
            except ImportError:
                logger.error("Pandas no disponible para procesar XLSX")
                return "", "error_no_pandas"

            # Descargar archivo Excel
            response = requests.get(xlsx_url)
            response.raise_for_status()
            excel_data = response.content

            # Leer todas las hojas
            excel_file = pd.ExcelFile(BytesIO(excel_data))
            all_text = ""

            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(BytesIO(excel_data), sheet_name=sheet_name)

                    if not df.empty:
                        all_text += f"\n[HOJA: {sheet_name}]\n"

                        # Extraer texto de todas las celdas
                        for column in df.columns:
                            all_text += f"\n[COLUMNA: {column}]\n"
                            for value in df[column].dropna():
                                if isinstance(value, (str, int, float)):
                                    all_text += f"{str(value)}\n"

                        all_text += "\n"

                except Exception as e:
                    logger.error(f"Error procesando hoja {sheet_name}: {e}")
                    continue

            return all_text, "xlsx_pandas_extraction"

        except Exception as e:
            logger.error(f"Error extrayendo texto de XLSX: {e}")
            return "", "error"

    def _find_text_matches_with_regex(
        self, text: str, keywords: List[str], original_phrase: str
    ) -> List[TextSearchMatch]:
        """
        Busca coincidencias en el texto usando regex para las palabras clave
        """
        try:
            matches = []
            text_lower = text.lower()

            # Dividir texto en páginas para obtener números de página
            pages = text.split("[PÁGINA")

            for page_idx, page_content in enumerate(pages):
                if page_idx == 0 and not page_content.strip().startswith("[PÁGINA"):
                    page_number = 1  # Primera página sin marcador
                    page_text = page_content
                elif page_content.strip():
                    # Extraer número de página
                    page_match = re.match(r"\s*(\d+)\]", page_content)
                    page_number = int(page_match.group(1)) if page_match else page_idx
                    page_text = (
                        page_content[page_content.find("]") + 1 :]
                        if "]" in page_content
                        else page_content
                    )
                else:
                    continue

                page_text_lower = page_text.lower()

                # Buscar frase completa primero
                phrase_pattern = re.escape(original_phrase.lower())
                for match in re.finditer(phrase_pattern, page_text_lower):
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(page_text), match.end() + 100)
                    context = page_text[context_start:context_end].strip()

                    matches.append(
                        TextSearchMatch(
                            page_number=page_number,
                            matched_text=original_phrase,
                            match_context=context,
                            confidence=1.0,  # Coincidencia exacta de frase
                            start_position=match.start(),
                            end_position=match.end(),
                        )
                    )

                # Buscar palabras clave individuales
                for keyword in keywords:
                    # Crear patrón regex para buscar la palabra completa
                    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

                    for match in re.finditer(pattern, page_text_lower):
                        # Extraer contexto alrededor de la coincidencia
                        context_start = max(0, match.start() - 100)
                        context_end = min(len(page_text), match.end() + 100)
                        context = page_text[context_start:context_end].strip()

                        # Calcular confianza basada en la longitud de la palabra
                        confidence = min(0.8, 0.4 + (len(keyword) * 0.1))

                        matches.append(
                            TextSearchMatch(
                                page_number=page_number,
                                matched_text=keyword,
                                match_context=context,
                                confidence=confidence,
                                start_position=match.start(),
                                end_position=match.end(),
                            )
                        )

            # Ordenar por confianza y eliminar duplicados
            unique_matches = []
            seen_contexts = set()

            for match in sorted(matches, key=lambda x: x.confidence, reverse=True):
                context_key = (
                    match.page_number,
                    match.matched_text.lower(),
                    match.match_context[:50],
                )
                if context_key not in seen_contexts:
                    unique_matches.append(match)
                    seen_contexts.add(context_key)

            return unique_matches[:20]  # Limitar a 20 coincidencias máximo

        except Exception as e:
            logger.error(f"Error buscando coincidencias de texto: {e}")
            return []

    async def _find_images_for_text_matches(
        self,
        file_url: str,
        text_matches: List[TextSearchMatch],
        search_phrase: str,
        file_type: str,
        max_results: int,
    ) -> List[ImageSearchResult]:
        """
        Encuentra imágenes relacionadas basándose en las coincidencias de texto
        """
        try:
            if file_type.lower() == "pdf":
                return await self._find_pdf_images_for_matches(
                    file_url, text_matches, search_phrase, max_results
                )
            elif file_type.lower() == "xlsx":
                return await self._find_xlsx_images_for_matches(
                    file_url, text_matches, search_phrase, max_results
                )
            else:
                return []

        except Exception as e:
            logger.error(f"Error buscando imágenes para coincidencias: {e}")
            return []

    async def _find_pdf_images_for_matches(
        self,
        pdf_url: str,
        text_matches: List[TextSearchMatch],
        search_phrase: str,
        max_results: int,
    ) -> List[ImageSearchResult]:
        """
        Busca imágenes en PDF basándose en coincidencias de texto
        """
        try:
            # Obtener páginas con coincidencias
            match_pages = list(set(match.page_number for match in text_matches))
            logger.info(
                f"🔍 Buscando imágenes en páginas con coincidencias: {match_pages}"
            )

            # Usar método existente de búsqueda con término
            result = await self.extract_text_and_images(
                pdf_url, search_phrase, "activate"
            )

            if (
                result
                and "related_images" in result
                and "images" in result["related_images"]
            ):
                all_images = result["related_images"]["images"]

                # Priorizar imágenes en páginas con coincidencias de texto
                prioritized_images = []
                other_images = []

                for img in all_images:
                    img_page = img.get("page", 0)
                    if img_page in match_pages:
                        # Aumentar confianza para imágenes en páginas con coincidencias
                        img["confidence"] = min(1.0, img.get("confidence", 0.5) + 0.3)
                        img["description"] += f" [En página con coincidencia de texto]"
                        prioritized_images.append(img)
                    else:
                        other_images.append(img)

                # Combinar imágenes priorizadas primero
                final_images = prioritized_images + other_images

                # Convertir a ImageSearchResult
                image_results = []
                for img in final_images[:max_results]:
                    image_results.append(
                        ImageSearchResult(
                            page_number=img.get("page", 0),
                            sheet_name=None,
                            image_base64=img.get("image_base64", ""),
                            description=img.get("description", ""),
                            confidence=img.get("confidence", 0.5),
                        )
                    )

                return image_results

            return []

        except Exception as e:
            logger.error(f"Error buscando imágenes en PDF: {e}")
            return []

    async def _find_xlsx_images_for_matches(
        self,
        xlsx_url: str,
        text_matches: List[TextSearchMatch],
        search_phrase: str,
        max_results: int,
    ) -> List[ImageSearchResult]:
        """
        Busca imágenes en XLSX basándose en coincidencias de texto
        """
        try:
            # Importar procesador de Excel
            from src.core.platform.fileprocessor.xlsx_processor import (
                ExcelImageProcessor,
            )

            excel_processor = ExcelImageProcessor(self.config_service)

            # Buscar imágenes usando el procesador de Excel
            excel_images = await excel_processor.search_images_in_excel(
                xlsx_url, search_phrase, max_results
            )

            # Aumentar confianza si hay coincidencias de texto
            if text_matches:
                for img in excel_images:
                    img.confidence = min(1.0, img.confidence + 0.2)
                    img.description += (
                        f" [Archivo con {len(text_matches)} coincidencias de texto]"
                    )

            return excel_images

        except Exception as e:
            logger.error(f"Error buscando imágenes en XLSX: {e}")
            return []

    async def _extract_text_with_ocr_comprehensive(self, pdf_data: bytes) -> str:
        """
        Extrae texto usando OCR de manera comprehensiva
        """
        try:
            all_text = ""

            # Intentar OCR en imágenes embebidas primero
            embedded_results = await self._extract_text_from_embedded_images(pdf_data)

            for result in embedded_results:
                all_text += f"\n[PÁGINA {result.page_number} - OCR EMBEBIDO]\n{result.extracted_text}\n"

            # Si no hay suficiente texto, usar OCR en páginas completas
            if len(all_text.strip()) < 100:
                page_results = await self._extract_text_from_full_pages(pdf_data)

                for result in page_results:
                    all_text += f"\n[PÁGINA {result.page_number} - OCR PÁGINA]\n{result.extracted_text}\n"

            return all_text

        except Exception as e:
            logger.error(f"Error en OCR comprehensivo: {e}")
            return ""

    async def _extract_text_and_image_positions_pdf(
        self, pdf_data: bytes
    ) -> Dict[str, Any]:
        """Extract text content and image positions from PDF file (simplified version)"""
        try:
            positions_data = {"page_text_content": {}}
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                positions_data["page_text_content"][page_num + 1] = [
                    {"text": page_text}
                ]

            pdf_document.close()
            return positions_data
        except Exception as e:
            logger.error(f"Error extracting PDF positions: {e}")
            return {"page_text_content": {}}

    def _analyze_image_with_python_tools(self, image_base64: str) -> Dict[str, Any]:
        """Placeholder for image analysis"""
        return {"quality_score": 0.7, "objects_detected": 1, "text_detected": False}

    def _create_detailed_description(self, technical_analysis: Dict[str, Any]) -> str:
        """Create a basic description from technical analysis"""
        return (
            f"Image analysis: quality {technical_analysis.get('quality_score', 0):.2f}"
        )

    async def _extract_all_images_aggressive(
        self, pdf_data: bytes, search_term: str = None
    ) -> List[Dict[str, Any]]:
        """
        Aggressive image extraction that finds ALL images in PDF regardless of filters.
        This method is designed to maximize recall for product catalogs.
        """
        try:
            logger.info(
                f"DEBUG: AGGRESSIVE image extraction mode for term '{search_term or 'ALL'}'"
            )

            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            all_images = []

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                page_text = page.get_text()

                logger.info(
                    f"  AGGRESSIVE: Page {page_num + 1}: {len(image_list)} images, {len(page_text)} chars text"
                )

                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)

                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))

                            # Very permissive size filter - only skip tiny images
                            if pil_image.width < 30 or pil_image.height < 30:
                                logger.info(
                                    f"    SKIPPED tiny image: {pil_image.width}x{pil_image.height}"
                                )
                                pix = None
                                continue

                            # Convert to base64
                            buffered = io.BytesIO()
                            pil_image.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()

                            # Generate basic description without AI calls for speed
                            basic_description = f"Image {img_index + 1} from page {page_num + 1} ({pil_image.width}x{pil_image.height})"

                            # If search term provided, check for basic text match
                            relevance_score = 0.4  # Default relevance
                            match_reasons = ["aggressive_extraction"]

                            if search_term:
                                search_lower = search_term.lower()
                                page_text_lower = page_text.lower()

                                # Check if page contains search term
                                if search_lower in page_text_lower:
                                    relevance_score = 0.8
                                    match_reasons.append("page_contains_term")
                                    basic_description += (
                                        f" [Page contains '{search_term}']"
                                    )

                                # Check for similar words
                                search_words = search_lower.split()
                                for word in search_words:
                                    if len(word) > 2 and word in page_text_lower:
                                        relevance_score = max(relevance_score, 0.6)
                                        match_reasons.append("page_contains_word")
                                        break

                            image_data = {
                                "page": page_num + 1,
                                "image_base64": img_base64,
                                "description": f"[AGGRESSIVE] {basic_description}",
                                "confidence": 0.7,  # Default confidence for aggressive mode
                                "related_to_text": (
                                    page_text[:200]
                                    if page_text
                                    else "No text found on page"
                                ),
                                "search_relevance": relevance_score,
                                "match_reasons": match_reasons,
                                "extraction_method": "aggressive",
                            }

                            all_images.append(image_data)
                            logger.info(
                                f"    EXTRACTED: Image {img_index + 1} ({pil_image.width}x{pil_image.height}) - Relevance: {relevance_score}"
                            )

                        pix = None

                    except Exception as e:
                        logger.error(
                            f"    ERROR extracting image {img_index + 1} from page {page_num + 1}: {e}"
                        )
                        continue

            pdf_document.close()

            # Sort by relevance if search term provided
            if search_term:
                all_images.sort(key=lambda x: x["search_relevance"], reverse=True)

            logger.info(f"AGGRESSIVE EXTRACTION: Found {len(all_images)} total images")
            return all_images

        except Exception as e:
            logger.error(f"Error in aggressive image extraction: {e}")
            return []

    async def debug_pdf_images(self, pdf_url: str) -> Dict[str, Any]:
        """
        Debug method to get basic info about PDF images without any filtering
        """
        try:
            # Download PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_data = response.content

            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            debug_info = {
                "total_pages": len(pdf_document),
                "pages_with_images": [],
                "total_images": 0,
                "images_by_page": {},
                "sample_texts": [],
            }

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()
                page_text = page.get_text()

                if image_list:
                    debug_info["pages_with_images"].append(page_num + 1)
                    debug_info["images_by_page"][page_num + 1] = len(image_list)
                    debug_info["total_images"] += len(image_list)

                # Sample some text from each page
                if page_text and len(page_text.strip()) > 10:
                    debug_info["sample_texts"].append(
                        {
                            "page": page_num + 1,
                            "text_preview": page_text[:200],
                            "contains_sabanas": "sábana" in page_text.lower()
                            or "sabana" in page_text.lower(),
                        }
                    )

            pdf_document.close()
            return debug_info

        except Exception as e:
            return {"error": f"Debug error: {str(e)}"}

    async def _extract_and_analyze_text_for_search(
        self, pdf_data: bytes, search_term: str
    ) -> Dict[str, Any]:
        """
        Extrae TODO el texto del PDF y analiza las coincidencias con el término de búsqueda usando regex.
        Devuelve información detallada sobre dónde se encontraron las palabras.
        """
        try:
            logger.info(
                f"DEBUG: Extrayendo texto completo del PDF para buscar '{search_term}'"
            )

            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            text_analysis = {
                "full_text": "",
                "pages_with_matches": [],
                "search_words": [],
                "regex_patterns": [],
                "total_matches": 0,
                "match_positions": [],
            }

            # Preparar las palabras de búsqueda y patrones regex
            search_term_clean = search_term.strip()

            # Extraer palabras individuales del término de búsqueda
            words = re.findall(r"\b\w+\b", search_term_clean.lower())
            text_analysis["search_words"] = words

            # Crear patrones regex flexibles para cada palabra
            regex_patterns = []
            for word in words:
                if len(word) >= 3:  # Solo palabras de 3+ caracteres
                    # Patrón que incluye variaciones (plurales, géneros, acentos)
                    word_pattern = self._create_flexible_word_pattern(word)
                    regex_patterns.append(word_pattern)

            text_analysis["regex_patterns"] = regex_patterns

            logger.info(f"  Palabras de búsqueda: {words}")
            logger.info(f"  Patrones regex creados: {len(regex_patterns)}")

            # Extraer texto de cada página y buscar coincidencias
            all_page_texts = []

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()

                if page_text.strip():
                    all_page_texts.append(
                        {"page": page_num + 1, "text": page_text, "matches": []}
                    )

                    # Buscar coincidencias usando regex en esta página
                    page_matches = self._find_regex_matches_in_text(
                        page_text, regex_patterns, words
                    )

                    if page_matches:
                        text_analysis["pages_with_matches"].append(
                            {
                                "page": page_num + 1,
                                "matches": page_matches,
                                "match_count": len(page_matches),
                                "text_snippet": (
                                    page_text[:300] + "..."
                                    if len(page_text) > 300
                                    else page_text
                                ),
                            }
                        )
                        text_analysis["total_matches"] += len(page_matches)
                        text_analysis["match_positions"].extend(page_matches)

                        logger.info(
                            f"  Página {page_num + 1}: {len(page_matches)} coincidencias encontradas"
                        )

            # Combinar todo el texto
            text_analysis["full_text"] = "\n\n".join(
                [p["text"] for p in all_page_texts]
            )

            pdf_document.close()

            logger.info(
                f"DEBUG: Análisis completado - {text_analysis['total_matches']} coincidencias en {len(text_analysis['pages_with_matches'])} páginas"
            )

            return text_analysis

        except Exception as e:
            logger.error(f"Error en análisis de texto: {e}")
            return {
                "full_text": "",
                "pages_with_matches": [],
                "search_words": [],
                "regex_patterns": [],
                "total_matches": 0,
                "match_positions": [],
            }

    def _create_flexible_word_pattern(self, word: str) -> str:
        """
        Crea un patrón regex flexible para una palabra que incluye variaciones comunes
        """
        try:
            # Escapar caracteres especiales
            escaped_word = re.escape(word.lower())

            # Crear patrón base
            base_pattern = escaped_word

            # Agregar variaciones para acentos comunes en español
            accent_map = {
                "a": "[aáàä]",
                "e": "[eéèë]",
                "i": "[iíìï]",
                "o": "[oóòö]",
                "u": "[uúùü]",
                "n": "[nñ]",
            }

            # Aplicar mapeo de acentos
            for base_char, pattern in accent_map.items():
                base_pattern = base_pattern.replace(base_char, pattern)

            # Crear patrón final que incluye:
            # - La palabra exacta
            # - Variaciones de plural/singular (s, es, as, os)
            # - Variaciones de género (a/o al final)
            flexible_pattern = rf"\b({base_pattern}[aeos]?s?)\b"

            return flexible_pattern

        except Exception as e:
            logger.error(f"Error creando patrón para '{word}': {e}")
            return rf"\b{re.escape(word)}\b"

    def _find_regex_matches_in_text(
        self, text: str, regex_patterns: List[str], original_words: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Encuentra todas las coincidencias usando patrones regex en el texto
        """
        try:
            matches = []
            text_lower = text.lower()

            for i, pattern in enumerate(regex_patterns):
                original_word = (
                    original_words[i] if i < len(original_words) else "unknown"
                )

                try:
                    # Buscar todas las coincidencias del patrón
                    pattern_matches = re.finditer(pattern, text_lower, re.IGNORECASE)

                    for match in pattern_matches:
                        match_info = {
                            "original_word": original_word,
                            "matched_text": match.group(),
                            "start_pos": match.start(),
                            "end_pos": match.end(),
                            "context_before": text[
                                max(0, match.start() - 50) : match.start()
                            ],
                            "context_after": text[match.end() : match.end() + 50],
                            "pattern_used": pattern,
                        }
                        matches.append(match_info)

                except re.error as regex_error:
                    logger.error(f"Error en patrón regex '{pattern}': {regex_error}")
                    # Fallback a búsqueda simple
                    if original_word.lower() in text_lower:
                        simple_matches = []
                        start = 0
                        while True:
                            pos = text_lower.find(original_word.lower(), start)
                            if pos == -1:
                                break
                            simple_matches.append(
                                {
                                    "original_word": original_word,
                                    "matched_text": original_word,
                                    "start_pos": pos,
                                    "end_pos": pos + len(original_word),
                                    "context_before": text[max(0, pos - 50) : pos],
                                    "context_after": text[
                                        pos
                                        + len(original_word) : pos
                                        + len(original_word)
                                        + 50
                                    ],
                                    "pattern_used": "simple_fallback",
                                }
                            )
                            start = pos + 1
                        matches.extend(simple_matches)

            return matches

        except Exception as e:
            logger.error(f"Error buscando coincidencias regex: {e}")
            return []

    async def _search_images_based_on_text_matches(
        self, pdf_data: bytes, text_analysis: Dict[str, Any], search_term: str
    ) -> List[Dict[str, Any]]:
        """
        Busca imágenes basadas en las coincidencias de texto encontradas con regex
        """
        try:
            logger.info(
                f"DEBUG: Buscando imágenes basadas en {text_analysis['total_matches']} coincidencias de texto"
            )

            if text_analysis["total_matches"] == 0:
                logger.info(
                    "  No se encontraron coincidencias de texto, no se pueden buscar imágenes relacionadas"
                )
                return []

            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            related_images = []

            # Procesar cada página que tiene coincidencias
            for page_info in text_analysis["pages_with_matches"]:
                page_num = page_info["page"]

                page_matches = page_info["matches"]

                logger.info(
                    f"  Procesando página {page_num} con {len(page_matches)} coincidencias..."
                )

                try:
                    page = pdf_document[page_num - 1]  # Convertir a índice 0
                    image_list = page.get_images()

                    logger.info(
                        f"    Encontradas {len(image_list)} imágenes en página {page_num}"
                    )

                    for img_index, img in enumerate(image_list):
                        try:
                            # Extraer imagen
                            xref = img[0]
                            pix = fitz.Pixmap(pdf_document, xref)

                            if pix.n - pix.alpha < 4:  # GRAY o RGB
                                img_data = pix.tobytes("png")
                                pil_image = Image.open(io.BytesIO(img_data))

                                # Filtrar imágenes muy pequeñas
                                if pil_image.width < 50 or pil_image.height < 50:
                                    pix = None
                                    continue

                                # Convertir a base64
                                buffered = io.BytesIO()
                                pil_image.save(buffered, format="PNG")
                                img_base64 = base64.b64encode(
                                    buffered.getvalue()
                                ).decode()

                                # Crear descripción contextual basada en las coincidencias
                                contextual_description = (
                                    self._create_contextual_description(
                                        page_matches,
                                        search_term,
                                        page_info.get("text_snippet", ""),
                                    )
                                )

                                # Calcular relevancia basada en proximidad a coincidencias de texto
                                relevance_score = self._calculate_text_match_relevance(
                                    page_matches,
                                    len(page_matches),
                                    text_analysis["total_matches"],
                                )

                                image_data = {
                                    "page": page_num,
                                    "image_base64": img_base64,
                                    "description": f"[REGEX MATCH] {contextual_description}",
                                    "confidence": min(
                                        0.8 + relevance_score, 1.0
                                    ),  # Alta confianza porque hay coincidencias de texto
                                    "related_to_text": f"Page contains {len(page_matches)} matches for '{search_term}': {page_info.get('text_snippet', '')[:150]}...",
                                    "search_relevance": relevance_score,
                                    "match_reasons": ["regex_text_match", "same_page"],
                                    "text_matches": page_matches,
                                    "match_count": len(page_matches),
                                }

                                related_images.append(image_data)
                                logger.info(
                                    f"      Imagen {img_index + 1}: Relevancia {relevance_score:.3f}, Confianza {image_data['confidence']:.3f}"
                                )

                            pix = None

                        except Exception as e:
                            logger.error(
                                f"      Error procesando imagen {img_index + 1}: {e}"
                            )
                            continue

                except Exception as e:
                    logger.error(f"    Error procesando página {page_num}: {e}")
                    continue

            pdf_document.close()

            # Ordenar por relevancia (más coincidencias = mayor relevancia)
            related_images.sort(
                key=lambda x: (x.get("match_count", 0), x.get("confidence", 0)),
                reverse=True,
            )

            logger.info(
                f"DEBUG: Búsqueda basada en texto completada. {len(related_images)} imágenes encontradas"
            )
            return related_images

        except Exception as e:
            logger.error(f"Error en búsqueda basada en coincidencias de texto: {e}")
            return []

    def _create_contextual_description(
        self, page_matches: List[Dict], search_term: str, page_text: str
    ) -> str:
        """
        Crea una descripción contextual basada en las coincidencias encontradas
        """
        try:
            matched_words = [match["matched_text"] for match in page_matches]
            unique_matches = list(set(matched_words))

            description = f"Image on page containing text matches for '{search_term}': {', '.join(unique_matches[:5])}"

            if len(page_matches) > 1:
                description += f" ({len(page_matches)} total matches found)"

            return description

        except Exception:
            return f"Image related to search term '{search_term}'"

    def _calculate_text_match_relevance(
        self, page_matches: List[Dict], page_match_count: int, total_matches: int
    ) -> float:
        """
        Calcula la relevancia basada en la cantidad y calidad de coincidencias de texto
        """
        try:
            if total_matches == 0:
                return 0.0

            # Relevancia base por tener coincidencias en la página
            base_relevance = 0.3

            # Bonus por cantidad de coincidencias en la página
            match_density = min(page_match_count / max(total_matches, 1), 1.0)
            density_bonus = match_density * 0.4

            # Bonus por múltiples coincidencias en la misma página
            if page_match_count > 1:
                multiple_bonus = min(page_match_count * 0.1, 0.3)
            else:
                multiple_bonus = 0.0

            total_relevance = base_relevance + density_bonus + multiple_bonus

            return min(max(total_relevance, 0.0), 1.0)

        except Exception:
            return 0.5

    # Ejemplo de uso de la nueva funcionalidad
    async def demo_smart_search(self, file_url: str, file_type: str = "pdf"):
        """
        Función de demostración para mostrar cómo usar la búsqueda inteligente
        """
        logger.info("🚀 DEMO: Búsqueda Inteligente con Análisis de Texto y Regex")
        logger.info("=" * 60)

        # Ejemplos de frases de búsqueda
        search_phrases = [
            "productos lácteos",
            "teléfonos celular smartphone",
            "computadora laptop notebook",
            "zapatos deportivos",
            "medicamentos pastillas",
        ]

        for phrase in search_phrases:
            logger.info(f"\n🔍 Buscando: '{phrase}'")
            logger.info("-" * 40)

            result = await self.smart_search_with_text_analysis(
                file_url=file_url,
                search_phrase=phrase,
                file_type=file_type,
                max_results=5,
            )

            logger.info(f"📝 Palabras clave extraídas: {result.processed_keywords}")
            logger.info(
                f"🎯 Coincidencias de texto encontradas: {result.total_matches}"
            )
            logger.info(f"📄 Método de extracción: {result.extraction_method}")
            logger.info(f"🖼️ Imágenes relacionadas: {len(result.related_images)}")

            # Mostrar algunas coincidencias de texto
            if result.text_matches:
                logger.info("\n📋 Primeras coincidencias de texto:")
                for i, match in enumerate(result.text_matches[:3]):
                    logger.info(
                        f"  {i+1}. Página {match.page_number}: '{match.matched_text}' "
                        f"(confianza: {match.confidence:.2f})"
                    )
                    logger.info(f"     Contexto: {match.match_context[:100]}...")

            # Mostrar información de imágenes
            if result.related_images:
                logger.info("\n🖼️ Imágenes encontradas:")
                for i, img in enumerate(result.related_images[:3]):
                    logger.info(
                        f"  {i+1}. Página {img.page_number}: {img.description[:100]}... "
                        f"(confianza: {img.confidence:.2f})"
                    )

            logger.info("\n" + "=" * 60)

        return "Demo completada"

    async def search_images_in_pdf(
        self, pdf_url: str, search_term: str, max_results: int = 5
    ):
        """
        Public method to search for images in a PDF file using a search term.
        Returns a list of ImageSearchResult.
        """
        result = await self.extract_text_and_images(pdf_url, search_term)
        images = result.get("related_images", {}).get("images", [])
        # If images are dicts, convert to ImageSearchResult
        image_results = []
        for img in images:
            if isinstance(img, dict):
                image_results.append(
                    ImageSearchResult(
                        page_number=img.get("page", 0),
                        sheet_name=None,
                        image_base64=img.get("image_base64", ""),
                        description=img.get("description", ""),
                        confidence=img.get("confidence", 0.0),
                    )
                )
            else:
                image_results.append(img)
        image_results = sorted(
            image_results, key=lambda x: getattr(x, "confidence", 0), reverse=True
        )
        return image_results[:max_results]
