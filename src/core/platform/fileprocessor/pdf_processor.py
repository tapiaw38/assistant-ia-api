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


@dataclass
class TextExtractionResult:
    """Text extraction result with related images"""

    extracted_text: str
    related_images: List[ImageSearchResult]
    text_confidence: float
    page_number: int
    extraction_method: str  # 'pdf_text', 'ocr_embedded', 'ocr_page'


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
                print("Text extracted from PDF successfully.")
                processing_method = "pdf_text_extraction"
            else:
                # If no text, use OCR on images
                text_extraction_results = await self._extract_text_with_ocr(pdf_data)
                print("Text extracted using OCR.")
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
            print(f"Error checking PDF text: {e}")
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
            print(f"Error extracting text from PDF: {e}")
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
            print(f"Error extracting text with OCR: {e}")
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
                        print(
                            f"Error processing embedded image {img_index} on page {page_num + 1}: {e}"
                        )
                        continue

            pdf_document.close()
            return results

        except Exception as e:
            print(f"Error extracting text from embedded images: {e}")
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
                    print(f"Error processing page {page_num + 1}: {e}")
                    continue

            return results

        except Exception as e:
            print(f"Error extracting text from full pages: {e}")
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
            print(f"Error with Tesseract OCR: {e}")
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
            print(f"Error preprocessing image: {e}")
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
            print(f"Error cleaning text: {e}")
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
                        relevance_score = await self._calculate_image_text_relevance(
                            img_base64, text
                        )

                        if relevance_score > 0.5:  # Increased relevance threshold for better precision
                            description = (
                                await self._generate_image_description_with_context(
                                    img_base64, text
                                )
                            )

                            image_result = ImageSearchResult(
                                page_number=page_number,
                                image_base64=img_base64,
                                description=description,
                                confidence=relevance_score,
                            )
                            related_images.append(image_result)

                    pix = None

                except Exception as e:
                    print(
                        f"Error processing image {img_index} on page {page_number}: {e}"
                    )
                    continue

            pdf_document.close()

            # Sort by relevance
            related_images.sort(key=lambda x: x.confidence, reverse=True)

            return related_images[:5]  # Maximum 5 related images

        except Exception as e:
            print(f"Error finding related images: {e}")
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
            print(f"Error calculating image-text relevance: {e}")
            return 0.4

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
            print(f"Error extracting keywords: {e}")
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
            print(f"Error in semantic analysis: {e}")
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
            print(f"Error generating enhanced description: {e}")
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
            print(f"Error identifying image pages: {e}")
            return []

    async def extract_text_and_images(
        self, pdf_url: str, search_term: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to extract text and images from a PDF.
        If the PDF has no text, uses OCR. Also identifies related images.

        Args:
            pdf_url: URL of the PDF to process
            search_term: Optional search term to filter images

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
        Filters images based on a search term
        """
        try:
            filtered = []
            search_term_lower = search_term.lower()

            for img in images:
                # Check in description
                description_match = search_term_lower in img["description"].lower()

                # Check in related text
                text_match = search_term_lower in img["related_to_text"].lower()

                # Additional semantic analysis
                semantic_score = await self._calculate_search_relevance(
                    img["description"], search_term
                )

                if description_match or text_match or semantic_score > 0.6:
                    # Adjust confidence based on search relevance
                    if description_match or text_match:
                        img["confidence"] = min(img["confidence"] + 0.2, 1.0)

                    img["search_relevance"] = semantic_score
                    
                    # Only include images with >75% confidence for better precision
                    if img["confidence"] > 0.75:
                        filtered.append(img)

            # Sort by search relevance and confidence
            filtered.sort(
                key=lambda x: (x.get("search_relevance", 0), x["confidence"]),
                reverse=True,
            )

            return filtered

        except Exception as e:
            print(f"Error filtering images: {e}")
            return images

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
            print(f"Error calculating search relevance: {e}")
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
            print(f"Error creating summary: {e}")
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
            print(f"Error generating recommendations: {e}")
            return ["Processing completed. Manual review recommended."]

    # Convenience method for compatibility with existing code
    async def search_images_in_pdf_with_text_analysis(
        self, pdf_url: str, search_term: str, max_results: int = 5
    ) -> List[ImageSearchResult]:
        """
        Enhanced search that considers both extracted text and images
        """
        try:
            # Use the new main method
            result = await self.extract_text_and_images(pdf_url, search_term)

            # Convert result to expected format
            image_results = []
            for img_data in result["related_images"]["images"][:max_results]:
                image_result = ImageSearchResult(
                    page_number=img_data["page"],
                    image_base64=img_data["image_base64"],
                    description=img_data["description"],
                    confidence=img_data["confidence"],
                )
                image_results.append(image_result)

            return image_results

        except Exception as e:
            print(f"Error in enhanced image search: {e}")
            # Fallback to original method
            return await self.search_images_in_pdf(pdf_url, search_term, max_results)

    def _analyze_image_with_python_tools(self, image_base64: str) -> Dict[str, Any]:
        """Image analysis using Python tools"""
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            img_array = np.array(image)

            analysis = {
                "dimensions": image.size,
                "mode": image.mode,
                "has_transparency": image.mode in ("RGBA", "LA")
                or "transparency" in image.info,
                "dominant_colors": self._get_dominant_colors(img_array),
                "brightness": self._calculate_brightness(img_array),
                "contrast": self._calculate_contrast(img_array),
                "sharpness": self._calculate_sharpness(img_array),
                "text_detected": self._detect_text_regions(img_array),
                "objects_detected": self._detect_objects(img_array),
                "quality_score": self._calculate_quality_score(img_array),
            }

            return analysis

        except Exception as e:
            return {"error": f"Error analyzing image: {str(e)}"}

    def _get_dominant_colors(self, img_array: np.ndarray) -> List[List[int]]:
        """Gets dominant colors from an image"""
        try:
            pixels = (
                img_array.reshape(-1, 3)
                if len(img_array.shape) == 3
                else img_array.reshape(-1, 1)
            )

            try:
                from sklearn.cluster import KMeans

                sample_size = min(1000, len(pixels))
                sample_pixels = pixels[:: len(pixels) // sample_size]

                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                kmeans.fit(sample_pixels)

                colors = kmeans.cluster_centers_.astype(int)
                return colors.tolist()

            except ImportError:
                return self._get_dominant_colors_fallback(img_array)
        except Exception:
            return []

    def _get_dominant_colors_fallback(self, img_array: np.ndarray) -> List[List[int]]:
        """Fallback method to get dominant colors without sklearn"""
        try:
            # Convert to PIL Image if necessary
            if isinstance(img_array, np.ndarray):
                if len(img_array.shape) == 3:
                    image = Image.fromarray(img_array)
                else:
                    image = Image.fromarray(img_array, mode="L")
            else:
                image = img_array

            # Reduce to color palette
            image_reduced = image.quantize(colors=5)
            palette = image_reduced.getpalette()

            # Extract dominant colors
            colors = []
            for i in range(5):
                color = palette[i * 3 : (i + 1) * 3]
                colors.append(color)

            return colors
        except Exception:
            return []

    def _calculate_brightness(self, img_array: np.ndarray) -> float:
        """Calculates average brightness of an image"""
        try:
            if len(img_array.shape) == 3:
                # Convert to grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            return float(np.mean(gray))
        except Exception:
            # Alternative method without OpenCV
            if len(img_array.shape) == 3:
                # Manual grayscale conversion
                gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array

            return float(np.mean(gray))

    def _calculate_contrast(self, img_array: np.ndarray) -> float:
        """Calculates image contrast"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            return float(np.std(gray))
        except Exception:
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array

            return float(np.std(gray))

    def _calculate_sharpness(self, img_array: np.ndarray) -> float:
        """Calculates image sharpness using Laplacian operator"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Apply Laplacian operator
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        except Exception:
            # Alternative method without OpenCV
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array

            # Manual Laplacian operator
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            try:
                laplacian = cv2.filter2D(gray, cv2.CV_64F, kernel)
                return float(laplacian.var())
            except:
                # Basic manual implementation if cv2 is not available
                return float(np.var(gray))

    def _detect_text_regions(self, img_array: np.ndarray) -> bool:
        """Detects if there are text regions in the image"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Apply edge detection filter
            edges = cv2.Canny(gray, 50, 150)

            # Find rectangular contours (possible text)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Count contours that could be text
            text_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Filter by area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 5:  # Typical text proportion
                        text_contours += 1

            return text_contours > 5

        except Exception:
            return False

    def _detect_objects(self, img_array: np.ndarray) -> int:
        """Detects approximate number of objects in the image"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter contours by area
            significant_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small objects
                    significant_contours += 1

            return significant_contours

        except Exception:
            return 0

    def _calculate_quality_score(self, img_array: np.ndarray) -> float:
        """Calculates image quality score based on various factors"""
        try:
            brightness = self._calculate_brightness(img_array)
            contrast = self._calculate_contrast(img_array)
            sharpness = self._calculate_sharpness(img_array)

            # Normalize values
            brightness_score = (
                1.0 - abs(brightness - 127.5) / 127.5
            )  # Optimal around 127.5
            contrast_score = min(contrast / 50.0, 1.0)  # High contrast is better
            sharpness_score = min(sharpness / 1000.0, 1.0)  # High sharpness is better

            # Calculate average score
            quality_score = (brightness_score + contrast_score + sharpness_score) / 3.0

            return float(quality_score)

        except Exception:
            return 0.5  # Neutral score if error

    def _create_detailed_description(self, analysis: Dict[str, Any]) -> str:
        """Creates detailed description based on technical analysis"""
        try:
            parts = []

            # Basic information
            dimensions = analysis.get("dimensions", (0, 0))
            parts.append(
                f"This is an image of {dimensions[0]} x {dimensions[1]} pixels"
            )

            # Overall quality
            quality_score = analysis.get("quality_score", 0.0)
            if quality_score > 0.8:
                parts.append("of excellent visual quality")
            elif quality_score > 0.6:
                parts.append("of good quality")
            elif quality_score > 0.4:
                parts.append("of medium quality")
            else:
                parts.append("of basic quality")

            # Technical characteristics
            brightness = analysis.get("brightness", 0)
            contrast = analysis.get("contrast", 0)
            sharpness = analysis.get("sharpness", 0)

            # Brightness
            if brightness > 200:
                parts.append("with very bright lighting")
            elif brightness > 150:
                parts.append("with good lighting")
            elif brightness > 100:
                parts.append("with moderate lighting")
            else:
                parts.append("with dim lighting")

            # Contrast
            if contrast > 70:
                parts.append("and high contrast")
            elif contrast > 40:
                parts.append("and moderate contrast")
            else:
                parts.append("and low contrast")

            # Sharpness
            if sharpness > 1000:
                parts.append("The image shows high sharpness")
            elif sharpness > 500:
                parts.append("The image has medium sharpness")
            else:
                parts.append("The image shows low sharpness")

            # Detected content
            objects_detected = analysis.get("objects_detected", 0)
            if objects_detected > 5:
                parts.append(
                    f"Multiple objects detected ({objects_detected}), suggesting a complex composition"
                )
            elif objects_detected > 2:
                parts.append(f"{objects_detected} main objects detected")
            elif objects_detected > 0:
                parts.append(f"{objects_detected} main object detected")
            else:
                parts.append("No clearly defined objects detected")

            # Text
            if analysis.get("text_detected", False):
                parts.append("The image contains text or textual elements")

            # Colors
            dominant_colors = analysis.get("dominant_colors", [])
            if len(dominant_colors) > 0:
                color_description = self._describe_colors(dominant_colors)
                parts.append(f"The predominant colors are: {color_description}")

            return ". ".join(parts) + "."

        except Exception as e:
            return f"Technical image analysis available. Error in detailed description: {str(e)}"

    def _describe_colors(self, colors: List[List[int]]) -> str:
        """Describes colors in a friendly way"""
        try:
            color_names = []
            for color in colors[:3]:  # Only first 3 colors
                r, g, b = color[:3]

                # Basic color classification
                if r > 200 and g > 200 and b > 200:
                    color_names.append("white")
                elif r < 50 and g < 50 and b < 50:
                    color_names.append("black")
                elif r > 200 and g < 100 and b < 100:
                    color_names.append("red")
                elif r < 100 and g > 200 and b < 100:
                    color_names.append("green")
                elif r < 100 and g < 100 and b > 200:
                    color_names.append("blue")
                elif r > 200 and g > 200 and b < 100:
                    color_names.append("yellow")
                elif r > 200 and g < 200 and b > 200:
                    color_names.append("magenta")
                elif r < 200 and g > 200 and b > 200:
                    color_names.append("cyan")
                elif r > 150 and g > 100 and b < 100:
                    color_names.append("brown")
                elif r > 150 and g > 150 and b > 150:
                    color_names.append("light gray")
                elif r < 150 and g < 150 and b < 150:
                    color_names.append("dark gray")
                else:
                    color_names.append("mixed color")

            return ", ".join(color_names)

        except Exception:
            return "varied colors"

    # Keep original methods for compatibility
    async def search_images_in_pdf(
        self, pdf_url: str, search_term: str, max_results: int = 5
    ) -> List[ImageSearchResult]:
        """
        Original PDF image search method (kept for compatibility)
        """
        try:
            # Use the new enhanced method
            return await self.search_images_in_pdf_with_text_analysis(
                pdf_url, search_term, max_results
            )
        except Exception as e:
            print(f"Error in image search: {e}")
            return []
