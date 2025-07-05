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
from src.core.platform.fileprocessor.processor import (
    BaseFileProcessor,
    ImageSearchResult,
)


class FileImageProcessor(BaseFileProcessor):
    """PDF file processor that extends base functionality"""

    def __init__(self, config_service: ConfigurationService):
        super().__init__(config_service)

    def _analyze_image_with_python_tools(self, image_base64: str) -> Dict[str, Any]:
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
        try:
            pixels = (
                img_array.reshape(-1, 3)
                if len(img_array.shape) == 3
                else img_array.reshape(-1, 1)
            )

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
        """Alternative method to get dominant colors without sklearn"""
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
        """Calculate average brightness of an image"""
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
        """Calculate contrast of an image"""
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
        """Calculate sharpness of an image using Laplacian operator"""
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
            laplacian = cv2.filter2D(gray, cv2.CV_64F, kernel)
            return float(laplacian.var())

    def _detect_text_regions(self, img_array: np.ndarray) -> bool:
        """Detect if there are text regions in the image"""
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
        """Detect approximate number of objects in the image"""
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
        """Calculate image quality score based on various factors"""
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
            return 0.5  # Neutral score if there's an error

    async def search_images_in_pdf(
        self, pdf_url: str, search_term: str, max_results: int = 5
    ) -> List[ImageSearchResult]:
        """
        Search for images in a PDF based on a search term

        Args:
            pdf_url: URL of the PDF to process
            search_term: Search term (e.g., "quilted")
            max_results: Maximum number of results

        Returns:
            List of image search results
        """
        try:
            # Download PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_data = response.content

            # Extract images from PDF
            images_data = await self._extract_images_from_pdf(pdf_data)

            # Search for relevant images using OpenAI Vision
            results = await self._search_relevant_images(
                images_data, search_term, max_results
            )

            return results

        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    async def _extract_images_from_pdf(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extract all images from a PDF"""
        images_data = []

        # Method 1: Use PyMuPDF to extract embedded images
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # Extract embedded images
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    # Get image object
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)

                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))

                        # Convert to base64
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()

                        images_data.append(
                            {
                                "page_number": page_num + 1,
                                "image_index": img_index,
                                "image_base64": img_base64,
                                "extraction_method": "embedded",
                            }
                        )

                    pix = None

                except Exception as e:
                    print(
                        f"Error extracting embedded image {img_index} from page {page_num + 1}: {e}"
                    )
                    continue

        pdf_document.close()

        # Method 2: Convert full pages to images (fallback)
        if len(images_data) == 0:
            images_data = await self._convert_pdf_pages_to_images(pdf_data)

        return images_data

    async def _convert_pdf_pages_to_images(
        self, pdf_data: bytes
    ) -> List[Dict[str, Any]]:
        """Convert full PDF pages to images"""
        images_data = []

        try:
            # Convert pages to images
            images = convert_from_bytes(pdf_data, dpi=200)

            for page_num, image in enumerate(images):
                # Convert to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                images_data.append(
                    {
                        "page_number": page_num + 1,
                        "image_index": 0,
                        "image_base64": img_base64,
                        "extraction_method": "page_conversion",
                    }
                )

        except Exception as e:
            raise Exception(f"Error converting pages to images: {e}")

        return images_data

    async def _search_relevant_images(
        self, images_data: List[Dict[str, Any]], search_term: str, max_results: int
    ) -> List[ImageSearchResult]:
        """Search for relevant images using Python analysis with improved ranking"""

        if not images_data:
            return []

        # Create tasks to process images in parallel (with limit)
        semaphore = asyncio.Semaphore(3)  # Limit to 3 simultaneous requests
        tasks = []

        for img_data in images_data:
            task = self._analyze_image_relevance(semaphore, img_data, search_term)
            tasks.append(task)

        # Execute analysis in parallel
        analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter and sort results with improved logic
        valid_results = []
        for i, result in enumerate(analysis_results):
            if isinstance(result, Exception):
                print(f"Error analyzing image {i}: {result}")
                continue

            if (
                result and result.confidence > 0.25
            ):  # Slightly lower threshold for more results
                valid_results.append(result)

        # Sort by confidence (descending) and then by page number (ascending) as tiebreaker
        valid_results.sort(key=lambda x: (-x.confidence, x.page_number))

        # Log the ranking for debugging
        print(f"PDF Images ranked by relevance for '{search_term}':")
        for i, result in enumerate(valid_results[:max_results]):
            print(
                f"  {i+1}. Page {result.page_number} - Confidence: {result.confidence:.3f}"
            )

        return valid_results[:max_results]

    async def _analyze_image_relevance(
        self, semaphore: asyncio.Semaphore, img_data: Dict[str, Any], search_term: str
    ) -> Optional[ImageSearchResult]:
        """
        Analyze an image to determine its relevance to the search term
        Uses only Python tools since DeepSeek doesn't support Vision API
        """

        async with semaphore:
            try:
                # Complete analysis with Python tools
                python_analysis = self._analyze_image_with_python_tools(
                    img_data["image_base64"]
                )

                # Relevance analysis based on technical characteristics
                relevance_score = self._calculate_relevance_score(
                    python_analysis, search_term
                )

                if relevance_score > 0.3:  # Confidence threshold
                    # Create description based on technical analysis
                    description = self._create_description_from_analysis(
                        python_analysis, search_term
                    )

                    return ImageSearchResult(
                        page_number=img_data["page_number"],
                        image_base64=img_data["image_base64"],
                        description=description,
                        confidence=relevance_score,
                    )

                return None

            except Exception as e:
                print(f"Error analyzing image: {e}")
                return None

    def _calculate_relevance_score(
        self, analysis: Dict[str, Any], search_term: str
    ) -> float:
        """
        Calculate relevance score based on technical characteristics
        """
        try:
            base_score = 0.0

            # Score based on image quality
            quality_score = analysis.get("quality_score", 0.0)
            if quality_score > 0.7:
                base_score += 0.3
            elif quality_score > 0.5:
                base_score += 0.2
            elif quality_score > 0.3:
                base_score += 0.1

            # Score based on object detection
            objects_detected = analysis.get("objects_detected", 0)
            if objects_detected > 2:
                base_score += 0.3
            elif objects_detected > 0:
                base_score += 0.2

            # Score based on detected text
            if analysis.get("text_detected", False):
                base_score += 0.2

            # Score based on colors (textile products usually have varied colors)
            dominant_colors = analysis.get("dominant_colors", [])
            if len(dominant_colors) > 2:
                base_score += 0.2

            # Bonus for specific searches
            search_term_lower = search_term.lower()

            # Specific heuristics for common terms
            if search_term_lower in [
                "quilted",
                "bedding",
                "comforter",
                "duvet",
                "quilt",
            ]:
                # For textiles, look for color and texture patterns
                if len(dominant_colors) > 1 and objects_detected > 0:
                    base_score += 0.3

            elif search_term_lower in ["furniture", "chair", "table", "sofa"]:
                # For furniture, look for geometric shapes
                if objects_detected > 0:
                    base_score += 0.4

            elif search_term_lower in ["product", "item", "article"]:
                # For generic products, any detected object is relevant
                if objects_detected > 0:
                    base_score += 0.5

            # Penalty for very low quality
            if quality_score < 0.2:
                base_score *= 0.5

            return min(base_score, 1.0)

        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0

    def _create_description_from_analysis(
        self, analysis: Dict[str, Any], search_term: str
    ) -> str:
        """
        Create description based on technical analysis
        """
        try:
            description_parts = []

            # Basic information
            dimensions = analysis.get("dimensions", (0, 0))
            description_parts.append(f"Image of {dimensions[0]}x{dimensions[1]} pixels")

            # Quality
            quality_score = analysis.get("quality_score", 0.0)
            if quality_score > 0.8:
                description_parts.append("of high quality")
            elif quality_score > 0.6:
                description_parts.append("of good quality")
            elif quality_score > 0.4:
                description_parts.append("of medium quality")
            else:
                description_parts.append("of low quality")

            # Detected objects
            objects_detected = analysis.get("objects_detected", 0)
            if objects_detected > 3:
                description_parts.append(
                    f"with multiple detected objects ({objects_detected})"
                )
            elif objects_detected > 1:
                description_parts.append(f"with {objects_detected} detected objects")
            elif objects_detected == 1:
                description_parts.append("with one main object")

            # Dominant colors
            dominant_colors = analysis.get("dominant_colors", [])
            if len(dominant_colors) > 0:
                color_description = self._describe_colors(dominant_colors)
                description_parts.append(
                    f"with predominant colors: {color_description}"
                )

            # Text
            if analysis.get("text_detected", False):
                description_parts.append("containing text")

            # Brightness and contrast
            brightness = analysis.get("brightness", 0)
            contrast = analysis.get("contrast", 0)

            if brightness > 200:
                description_parts.append("with bright lighting")
            elif brightness < 80:
                description_parts.append("with dim lighting")

            if contrast > 60:
                description_parts.append("and high contrast")
            elif contrast < 30:
                description_parts.append("and low contrast")

            # Relation to search term
            search_term_lower = search_term.lower()
            if search_term_lower in ["quilted", "bedding", "comforter", "duvet"]:
                description_parts.append(f"Possible content related to {search_term}")
            elif objects_detected > 0:
                description_parts.append(
                    f"Content potentially related to {search_term}"
                )

            return ". ".join(description_parts).capitalize() + "."

        except Exception as e:
            return f"Technically analyzed image. Possible relation to {search_term}."

    def _describe_colors(self, colors: List[List[int]]) -> str:
        """
        Describe colors in a user-friendly way
        """
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

    async def describe_image(self, image_base64: str) -> str:
        """
        Describe an image using technical analysis with Python tools
        since DeepSeek doesn't support Vision API
        """
        try:
            # Analysis with Python tools
            python_analysis = self._analyze_image_with_python_tools(image_base64)

            # Create detailed description based on technical analysis
            description = self._create_detailed_description(python_analysis)

            # Use DeepSeek to enhance description with technical context
            enhanced_description = await self._enhance_description_with_ai(
                description, python_analysis
            )

            return enhanced_description

        except Exception as e:
            return f"Error describing image: {str(e)}"

    def _create_detailed_description(self, analysis: Dict[str, Any]) -> str:
        """
        Create detailed description based on technical analysis
        """
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

            # Composition analysis
            if objects_detected > 0 and quality_score > 0.6:
                parts.append(
                    "The composition suggests it could be a product photograph or commercial content"
                )
            elif analysis.get("text_detected", False) and objects_detected > 0:
                parts.append(
                    "The combination of text and images suggests informational or advertising content"
                )

            return ". ".join(parts) + "."

        except Exception as e:
            return f"Technical image analysis available. Error in detailed description: {str(e)}"

    async def _enhance_description_with_ai(
        self, technical_description: str, analysis: Dict[str, Any]
    ) -> str:
        """
        Enhance technical description using DeepSeek with text only
        """
        try:
            prompt = f"""
            Based on this technical analysis of an image, create a more natural and understandable description:

            TECHNICAL ANALYSIS:
            {technical_description}

            SPECIFIC DATA:
            - Dimensions: {analysis.get('dimensions', 'N/A')}
            - Quality: {analysis.get('quality_score', 0):.2f}/1.0
            - Brightness: {analysis.get('brightness', 0):.1f}/255
            - Contrast: {analysis.get('contrast', 0):.1f}
            - Sharpness: {analysis.get('sharpness', 0):.1f}
            - Objects detected: {analysis.get('objects_detected', 0)}
            - Text detected: {'Yes' if analysis.get('text_detected', False) else 'No'}

            Instructions:
            1. Rewrite the description in a more natural and fluid way
            2. Keep technical information but make it more understandable
            3. Suggest what type of content it could be based on characteristics
            4. Respond in English
            5. Maximum 200 words
            """

            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
            )

            ai_description = response.choices[0].message.content.strip()

            # Combine AI description with technical data
            final_description = f"""{ai_description}

---
TECHNICAL ANALYSIS:
• Resolution: {analysis.get('dimensions', 'N/A')}
• Quality: {analysis.get('quality_score', 0):.2f}/1.0
• Brightness: {analysis.get('brightness', 0):.1f}/255
• Contrast: {analysis.get('contrast', 0):.1f}
• Sharpness: {'High' if analysis.get('sharpness', 0) > 1000 else 'Medium' if analysis.get('sharpness', 0) > 500 else 'Low'}
• Objects detected: {analysis.get('objects_detected', 0)}
• Contains text: {'Yes' if analysis.get('text_detected', False) else 'No'}"""

            return final_description

        except Exception as e:
            # If AI enhancement fails, return technical description
            return f"{technical_description}\n\n[Error enhancing description with AI: {str(e)}]"

    async def advanced_image_analysis(
        self, image_base64: str, analysis_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Advanced image analysis using Python tools and DeepSeek (text only)

        Args:
            image_base64: Image in base64
            analysis_type: Type of analysis ("general", "product", "document", "quality")

        Returns:
            Dictionary with complete analysis
        """
        try:
            # Analysis with Python tools
            python_analysis = self._analyze_image_with_python_tools(image_base64)

            # Create analysis based on technical characteristics
            ai_analysis = await self._create_ai_analysis_from_technical(
                python_analysis, analysis_type
            )

            # Combine analysis
            complete_analysis = {
                "ai_analysis": ai_analysis,
                "technical_analysis": python_analysis,
                "combined_quality_score": (
                    ai_analysis.get("confidence", 0.5)
                    + python_analysis.get("quality_score", 0.5)
                )
                / 2,
                "analysis_type": analysis_type,
            }

            return complete_analysis

        except Exception as e:
            return {"error": f"Error in advanced analysis: {str(e)}"}

    async def _create_ai_analysis_from_technical(
        self, python_analysis: Dict[str, Any], analysis_type: str
    ) -> Dict[str, Any]:
        """
        Create AI analysis based on technical data
        """
        try:
            # Create technical description
            technical_summary = self._create_detailed_description(python_analysis)

            # Specific prompts by type
            type_prompts = {
                "general": "Based on this technical analysis, describe the general elements of the image",
                "product": "Based on this technical analysis, evaluate if this could be a commercial product",
                "document": "Based on this technical analysis, evaluate if this is documentary content",
                "quality": "Based on this technical analysis, evaluate the overall quality of the image",
            }

            prompt = f"""
            {type_prompts.get(analysis_type, type_prompts["general"])}
            
            TECHNICAL ANALYSIS:
            {technical_summary}
            
            SPECIFIC DATA:
            - Dimensions: {python_analysis.get('dimensions', 'N/A')}
            - Quality: {python_analysis.get('quality_score', 0):.2f}/1.0
            - Objects detected: {python_analysis.get('objects_detected', 0)}
            - Text detected: {'Yes' if python_analysis.get('text_detected', False) else 'No'}
            - Dominant colors: {len(python_analysis.get('dominant_colors', []))} colors
            
            Respond in JSON with:
            {{
                "summary": "1-2 line summary",
                "elements": ["list", "of", "detected", "elements"],
                "colors": ["color", "descriptions"],
                "quality_assessment": "quality evaluation",
                "recommendations": ["recommendations", "based", "on", "analysis"],
                "confidence": 0.0-1.0
            }}
            """

            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON
            import json

            try:
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                json_str = content[start_idx:end_idx]
                ai_analysis = json.loads(json_str)
            except:
                # Fallback if JSON can't be parsed
                ai_analysis = {
                    "summary": "Technical analysis completed",
                    "elements": ["detected objects", "technical characteristics"],
                    "colors": [
                        self._describe_colors(
                            python_analysis.get("dominant_colors", [])
                        )
                    ],
                    "quality_assessment": f"Quality: {python_analysis.get('quality_score', 0):.2f}/1.0",
                    "recommendations": ["Image processed with technical tools"],
                    "confidence": python_analysis.get("quality_score", 0.5),
                }

            return ai_analysis

        except Exception as e:
            return {
                "summary": f"Error in AI analysis: {str(e)}",
                "elements": ["technical analysis available"],
                "colors": ["detected colors"],
                "quality_assessment": "Technical analysis completed",
                "recommendations": ["Review manually"],
                "confidence": 0.3,
            }

    def enhance_image_quality(self, image_base64: str) -> str:
        """
        Enhance image quality using Python tools

        Args:
            image_base64: Image in base64

        Returns:
            Enhanced image in base64
        """
        try:
            # Decode image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            # Apply enhancements
            enhanced = image

            # Adjust brightness and contrast
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.1)  # Slightly brighter

            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)  # More contrast

            # Improve sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)  # More sharpness

            # Apply smoothing filter if necessary
            enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)

            # Convert back to base64
            buffered = io.BytesIO()
            enhanced.save(buffered, format="PNG")
            enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()

            return enhanced_base64

        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image_base64  # Return original image if error
