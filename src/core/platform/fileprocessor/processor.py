import base64
import io
import requests
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter
from src.core.platform.config.service import ConfigurationService
from openai import AsyncOpenAI
import asyncio
from dataclasses import dataclass
import cv2
import numpy as np
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ImageSearchResult:
    page_number: int
    sheet_name: Optional[str]  # For Excel sheets
    image_base64: str
    description: str
    confidence: float
    bbox: Optional[Dict[str, float]] = None


@dataclass
class ProcessingResult:
    success: bool
    data: Any = None
    error: Optional[str] = None


class BaseFileProcessor:
    """Base class with common functionality for file processing"""

    def __init__(self, config_service: ConfigurationService):
        config = config_service.openai_config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.model = config.model

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _analyze_image_with_python_tools(self, image_base64: str) -> Dict[str, Any]:
        """Analyze image using Python tools (common for all processors)"""
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
        """Get dominant colors from image"""
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
        """Alternative method to get dominant colors without sklearn"""
        try:
            if isinstance(img_array, np.ndarray):
                if len(img_array.shape) == 3:
                    image = Image.fromarray(img_array)
                else:
                    image = Image.fromarray(img_array, mode="L")
            else:
                image = img_array

            image_reduced = image.quantize(colors=5)
            palette = image_reduced.getpalette()

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
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            return float(np.mean(gray))
        except Exception:
            if len(img_array.shape) == 3:
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

            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        except Exception:
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array

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

            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            text_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 5:
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

            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            significant_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
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

            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
            contrast_score = min(contrast / 50.0, 1.0)
            sharpness_score = min(sharpness / 1000.0, 1.0)

            quality_score = (brightness_score + contrast_score + sharpness_score) / 3.0
            return float(quality_score)
        except Exception:
            return 0.5

    def _calculate_relevance_score(
        self, analysis: Dict[str, Any], search_term: str
    ) -> float:
        """Calculate relevance score based on technical characteristics"""
        try:
            base_score = 0.0

            # Quality score contributes significantly to relevance
            quality_score = analysis.get("quality_score", 0.0)
            if quality_score > 0.8:
                base_score += 0.35  # High quality images are more relevant
            elif quality_score > 0.6:
                base_score += 0.25
            elif quality_score > 0.4:
                base_score += 0.15
            elif quality_score > 0.2:
                base_score += 0.05

            # Objects detected indicates content richness
            objects_detected = analysis.get("objects_detected", 0)
            if objects_detected > 5:
                base_score += 0.40  # Many objects = complex, detailed content
            elif objects_detected > 3:
                base_score += 0.30
            elif objects_detected > 1:
                base_score += 0.20
            elif objects_detected > 0:
                base_score += 0.10

            # Text detection adds context relevance
            if analysis.get("text_detected", False):
                base_score += 0.15

            # Color variety indicates visual richness
            dominant_colors = analysis.get("dominant_colors", [])
            if len(dominant_colors) > 4:
                base_score += 0.20
            elif len(dominant_colors) > 2:
                base_score += 0.15
            elif len(dominant_colors) > 1:
                base_score += 0.10

            # Search term specific bonuses
            search_term_lower = search_term.lower()

            # Generic product/content terms
            if search_term_lower in ["product", "item", "article", "object"]:
                if objects_detected > 0:
                    base_score += 0.30

            # Visual content terms
            elif search_term_lower in [
                "chart",
                "graph",
                "table",
                "diagram",
                "visualization",
            ]:
                if objects_detected > 0:
                    base_score += 0.25
                if analysis.get("text_detected", False):
                    base_score += 0.15

            # Image quality terms
            elif search_term_lower in ["image", "photo", "picture", "illustration"]:
                base_score += 0.20
                if quality_score > 0.6:
                    base_score += 0.15

            # Dimension-based relevance (larger images often more important)
            dimensions = analysis.get("dimensions", (0, 0))
            if dimensions[0] > 0 and dimensions[1] > 0:
                pixel_count = dimensions[0] * dimensions[1]
                if pixel_count > 500000:  # High resolution
                    base_score += 0.10
                elif pixel_count > 100000:  # Medium resolution
                    base_score += 0.05

            # Penalty for very low quality (likely corrupted or irrelevant)
            if quality_score < 0.1:
                base_score *= 0.3
            elif quality_score < 0.2:
                base_score *= 0.6

            # Ensure score is between 0 and 1
            return min(max(base_score, 0.0), 1.0)

        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0

    def _describe_colors(self, colors: List[List[int]]) -> str:
        """Describe colors in a user-friendly way"""
        try:
            color_names = []
            for color in colors[:3]:
                r, g, b = color[:3]

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
                else:
                    color_names.append("mixed color")

            return ", ".join(color_names)
        except Exception:
            return "varied colors"

    def _create_description_from_analysis(
        self, analysis: Dict[str, Any], search_term: str, context: str = ""
    ) -> str:
        """Create description based on technical analysis"""
        try:
            description_parts = []

            dimensions = analysis.get("dimensions", (0, 0))
            description_parts.append(f"Image of {dimensions[0]}x{dimensions[1]} pixels")

            quality_score = analysis.get("quality_score", 0.0)
            if quality_score > 0.8:
                description_parts.append("of high quality")
            elif quality_score > 0.6:
                description_parts.append("of good quality")
            elif quality_score > 0.4:
                description_parts.append("of medium quality")
            else:
                description_parts.append("of low quality")

            objects_detected = analysis.get("objects_detected", 0)
            if objects_detected > 3:
                description_parts.append(
                    f"with multiple detected objects ({objects_detected})"
                )
            elif objects_detected > 1:
                description_parts.append(f"with {objects_detected} detected objects")
            elif objects_detected == 1:
                description_parts.append("with one main object")

            dominant_colors = analysis.get("dominant_colors", [])
            if len(dominant_colors) > 0:
                color_description = self._describe_colors(dominant_colors)
                description_parts.append(
                    f"with predominant colors: {color_description}"
                )

            if analysis.get("text_detected", False):
                description_parts.append("containing text")

            if context:
                description_parts.append(f"from {context}")

            if objects_detected > 0:
                description_parts.append(
                    f"Content potentially related to {search_term}"
                )

            return ". ".join(description_parts).capitalize() + "."
        except Exception as e:
            return f"Technically analyzed image. Possible relation to {search_term}."

    async def describe_image(self, image_base64: str) -> str:
        """Describe an image using technical analysis"""
        try:
            python_analysis = self._analyze_image_with_python_tools(image_base64)
            description = self._create_detailed_description(python_analysis)
            enhanced_description = await self._enhance_description_with_ai(
                description, python_analysis
            )
            return enhanced_description
        except Exception as e:
            return f"Error describing image: {str(e)}"

    def _create_detailed_description(self, analysis: Dict[str, Any]) -> str:
        """Create detailed description based on technical analysis"""
        try:
            parts = []

            dimensions = analysis.get("dimensions", (0, 0))
            parts.append(
                f"This is an image of {dimensions[0]} x {dimensions[1]} pixels"
            )

            quality_score = analysis.get("quality_score", 0.0)
            if quality_score > 0.8:
                parts.append("of excellent visual quality")
            elif quality_score > 0.6:
                parts.append("of good quality")
            elif quality_score > 0.4:
                parts.append("of medium quality")
            else:
                parts.append("of basic quality")

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

            if analysis.get("text_detected", False):
                parts.append("The image contains text or textual elements")

            dominant_colors = analysis.get("dominant_colors", [])
            if len(dominant_colors) > 0:
                color_description = self._describe_colors(dominant_colors)
                parts.append(f"The predominant colors are: {color_description}")

            return ". ".join(parts) + "."
        except Exception as e:
            return f"Technical image analysis available. Error in detailed description: {str(e)}"

    async def _enhance_description_with_ai(
        self, technical_description: str, analysis: Dict[str, Any]
    ) -> str:
        """Enhance technical description using AI"""
        try:
            prompt = f"""
            Based on this technical analysis of an image, create a more natural and understandable description:

            TECHNICAL ANALYSIS:
            {technical_description}

            SPECIFIC DATA:
            - Dimensions: {analysis.get('dimensions', 'N/A')}
            - Quality: {analysis.get('quality_score', 0):.2f}/1.0
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
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
            )

            ai_description = response.choices[0].message.content.strip()

            final_description = f"""{ai_description}

---
TECHNICAL ANALYSIS:
• Resolution: {analysis.get('dimensions', 'N/A')}
• Quality: {analysis.get('quality_score', 0):.2f}/1.0
• Objects detected: {analysis.get('objects_detected', 0)}
• Contains text: {'Yes' if analysis.get('text_detected', False) else 'No'}"""

            return final_description
        except Exception as e:
            return f"{technical_description}\n\n[Error enhancing description with AI: {str(e)}]"

    def enhance_image_quality(self, image_base64: str) -> str:
        """Enhance image quality using Python tools"""
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            enhanced = image

            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.1)

            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)

            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)

            enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)

            buffered = io.BytesIO()
            enhanced.save(buffered, format="PNG")
            enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()

            return enhanced_base64
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image_base64
