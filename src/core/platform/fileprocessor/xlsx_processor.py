import base64
import io
import requests
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import asyncio
import zipfile
import xml.etree.ElementTree as ET
import re
from dataclasses import dataclass
from src.core.platform.fileprocessor.processor import (
    BaseFileProcessor,
    ImageSearchResult,
)
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Remove os import and environment variable usage
# Set module-level constants
IMAGE_CONFIDENCE_THRESHOLD = 0.3
MAX_IMAGES_TO_SHOW = 10


@dataclass
class ExcelTextSearchMatch:
    """Result of a text search match in Excel"""
    sheet_name: str
    cell_reference: str
    matched_text: str
    match_context: str
    confidence: float
    row_number: int
    column_name: str


@dataclass
class ExcelSmartSearchResult:
    """Result of smart search in Excel including text analysis and related images"""
    search_term: str
    processed_keywords: List[str]
    text_matches: List[ExcelTextSearchMatch]
    related_images: List[ImageSearchResult]
    total_matches: int
    extraction_method: str
    sheets_analyzed: List[str]


class ExcelImageProcessor(BaseFileProcessor):
    """Excel file processor that can extract and analyze embedded images and charts/graphs"""

    def __init__(self, config_service):
        super().__init__(config_service)
        # Enable fast processing to reduce API calls
        self.enable_fast_processing = True

    async def search_images_in_excel(
        self, excel_url: str, search_term: str, max_results: int = 5
    ) -> List[ImageSearchResult]:
        """
        Search for embedded images and charts/graphs in an Excel file based on a search term

        Args:
            excel_url: URL of the Excel file to process
            search_term: Search term (e.g., "product", "chart", "logo", "image")
            max_results: Maximum number of results

        Returns:
            List of image search results including both embedded images and generated charts
            Note: Only returns images with confidence > 65%
            Images closer to matching text (especially on the same X-axis/row) receive higher confidence scores
        """
        try:
            # Download Excel file
            response = requests.get(excel_url)
            response.raise_for_status()
            excel_data = response.content

            # Extract images from Excel (charts/graphs)
            images_data = await self._extract_images_from_excel(excel_data, excel_url)

            # Search for relevant images
            results = await self._search_relevant_images(
                images_data, search_term, max_results
            )

            return results

        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")

    async def _extract_images_from_excel(
        self, excel_data: bytes, file_url: str
    ) -> List[Dict[str, Any]]:
        """Extract charts and visualizations from Excel file"""
        images_data = []

        try:
            # Extract positions data for proximity analysis
            self.positions_data = await self._extract_text_and_image_positions(excel_data)
            
            # First, extract embedded images directly from the Excel file
            embedded_images = await self._extract_all_embedded_images(excel_data)
            images_data.extend(embedded_images)

            # Then, read Excel file with all sheets for data-driven charts
            excel_file = pd.ExcelFile(BytesIO(excel_data))

            for sheet_index, sheet_name in enumerate(excel_file.sheet_names):
                try:
                    # Read sheet data
                    df = pd.read_excel(BytesIO(excel_data), sheet_name=sheet_name)

                    # Skip empty sheets
                    if df.empty:
                        continue

                    # Generate visualizations from data
                    sheet_images = await self._generate_charts_from_sheet(
                        df, sheet_name, sheet_index
                    )
                    images_data.extend(sheet_images)

                except Exception as e:
                    logger.error(f"Error processing sheet {sheet_name}: {e}")
                    continue

            return images_data

        except Exception as e:
            raise Exception(f"Error extracting images from Excel: {e}")

    async def _extract_all_embedded_images(
        self, excel_data: bytes
    ) -> List[Dict[str, Any]]:
        """Extract all embedded images from Excel file"""
        embedded_images = []

        try:
            # Excel files are ZIP archives
            with zipfile.ZipFile(BytesIO(excel_data), "r") as zip_file:
                file_list = zip_file.namelist()

                # Extract images from media folder
                media_files = [f for f in file_list if f.startswith("xl/media/")]

                for media_file in media_files:
                    try:
                        image_data = zip_file.read(media_file)

                        if self._is_valid_image(image_data):
                            image_base64 = base64.b64encode(image_data).decode()

                            # Try to determine the sheet association
                            sheet_info = await self._get_image_sheet_association(
                                zip_file, media_file
                            )

                            embedded_images.append(
                                {
                                    "page_number": sheet_info.get("sheet_index", 1),
                                    "sheet_name": sheet_info.get(
                                        "sheet_name", "Unknown"
                                    ),
                                    "image_base64": image_base64,
                                    "extraction_method": "embedded_image_extracted",
                                    "media_file": media_file,
                                    "image_type": "embedded",
                                }
                            )

                    except Exception as e:
                        logger.error(f"Error extracting media file {media_file}: {e}")
                        continue

            return embedded_images

        except Exception as e:
            logger.error(f"Error extracting embedded images: {e}")
            return []

    async def _get_image_sheet_association(
        self, zip_file: zipfile.ZipFile, media_file: str
    ) -> Dict[str, Any]:
        """Try to determine which sheet an image belongs to"""
        try:
            # This is a simplified approach
            # In a full implementation, we'd parse the relationships XML files
            # to properly map images to sheets

            # For now, we'll return default values
            return {"sheet_index": 1, "sheet_name": "Sheet1"}

        except Exception as e:
            logger.error(f"Error determining sheet association: {e}")
            return {"sheet_index": 1, "sheet_name": "Unknown"}

    async def _generate_charts_from_sheet(
        self, df: pd.DataFrame, sheet_name: str, sheet_index: int
    ) -> List[Dict[str, Any]]:
        """Generate charts from DataFrame data"""
        charts_data = []

        try:
            # Identify numeric columns for charting
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) == 0:
                return charts_data

            # Generate different types of charts
            chart_types = [
                ("bar", self._create_bar_chart),
                ("line", self._create_line_chart),
                ("scatter", self._create_scatter_chart),
                ("heatmap", self._create_heatmap),
            ]

            for chart_type, chart_function in chart_types:
                try:
                    chart_base64 = await chart_function(df, numeric_cols)
                    if chart_base64:
                        charts_data.append(
                            {
                                "page_number": sheet_index + 1,
                                "sheet_name": sheet_name,
                                "image_base64": chart_base64,
                                "extraction_method": f"{chart_type}_chart_generated",
                                "chart_type": chart_type,
                            }
                        )
                except Exception as e:
                    logger.error(
                        f"Error creating {chart_type} chart for sheet {sheet_name}: {e}"
                    )
                    continue

            return charts_data

        except Exception as e:
            logger.error(f"Error generating charts from sheet {sheet_name}: {e}")
            return []

    async def _create_bar_chart(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> Optional[str]:
        """Create bar chart from DataFrame"""
        try:
            plt.figure(figsize=(10, 6))

            # Use first few numeric columns
            cols_to_plot = numeric_cols[:3]

            if len(df) > 20:
                # Sample data if too large
                sample_df = df.head(20)
            else:
                sample_df = df

            # Create bar chart
            if len(cols_to_plot) == 1:
                sample_df[cols_to_plot[0]].plot(kind="bar")
            else:
                sample_df[cols_to_plot].plot(kind="bar")

            plt.title(f'Bar Chart - {", ".join(cols_to_plot)}')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()

            plt.close()
            return chart_base64

        except Exception as e:
            plt.close()
            logger.error(f"Error creating bar chart: {e}")
            return None

    async def _create_line_chart(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> Optional[str]:
        """Create line chart from DataFrame"""
        try:
            plt.figure(figsize=(10, 6))

            cols_to_plot = numeric_cols[:3]

            if len(df) > 50:
                sample_df = df.head(50)
            else:
                sample_df = df

            # Create line chart
            for col in cols_to_plot:
                plt.plot(sample_df.index, sample_df[col], label=col, marker="o")

            plt.title(f'Line Chart - {", ".join(cols_to_plot)}')
            plt.xlabel("Index")
            plt.ylabel("Values")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()

            plt.close()
            return chart_base64

        except Exception as e:
            plt.close()
            logger.error(f"Error creating line chart: {e}")
            return None

    async def _create_scatter_chart(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> Optional[str]:
        """Create scatter plot from DataFrame"""
        try:
            if len(numeric_cols) < 2:
                return None

            plt.figure(figsize=(10, 6))

            x_col = numeric_cols[0]
            y_col = numeric_cols[1]

            # Sample data if too large
            if len(df) > 1000:
                sample_df = df.sample(n=1000)
            else:
                sample_df = df

            plt.scatter(sample_df[x_col], sample_df[y_col], alpha=0.6)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"Scatter Plot - {x_col} vs {y_col}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()

            plt.close()
            return chart_base64

        except Exception as e:
            plt.close()
            logger.error(f"Error creating scatter chart: {e}")
            return None

    async def _create_heatmap(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> Optional[str]:
        """Create heatmap from DataFrame correlation"""
        try:
            if len(numeric_cols) < 2:
                return None

            plt.figure(figsize=(10, 8))

            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Create heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                fmt=".2f",
            )
            plt.title("Correlation Heatmap")
            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()

            plt.close()
            return chart_base64

        except Exception as e:
            plt.close()
            logger.error(f"Error creating heatmap: {e}")
            return None

    async def _extract_embedded_images_from_sheet(
        self, excel_data: bytes, sheet_name: str, sheet_index: int
    ) -> List[Dict[str, Any]]:
        """Extract embedded images from Excel sheet (charts, pictures)"""
        embedded_images = []

        try:
            # Excel files are essentially ZIP files, so we can extract them
            with zipfile.ZipFile(BytesIO(excel_data), "r") as zip_file:
                # Get list of all files in the Excel archive
                file_list = zip_file.namelist()

                # Look for media files (images)
                media_files = [f for f in file_list if f.startswith("xl/media/")]

                for media_file in media_files:
                    try:
                        # Extract the image file
                        image_data = zip_file.read(media_file)

                        # Check if it's a valid image
                        if self._is_valid_image(image_data):
                            # Convert to base64
                            image_base64 = base64.b64encode(image_data).decode()

                            # Try to determine which sheet this image belongs to
                            # This is a simplified approach - in reality, we'd need to parse
                            # the relationships and drawing XML files
                            embedded_images.append(
                                {
                                    "page_number": sheet_index + 1,
                                    "sheet_name": sheet_name,
                                    "image_base64": image_base64,
                                    "extraction_method": "embedded_image_extracted",
                                    "media_file": media_file,
                                    "image_type": "embedded",
                                }
                            )

                    except Exception as e:
                        logger.error(f"Error extracting media file {media_file}: {e}")
                        continue

                # Also look for chart images in drawings
                drawing_files = [f for f in file_list if f.startswith("xl/drawings/")]
                for drawing_file in drawing_files:
                    try:
                        # Parse drawing XML to find chart relationships
                        drawing_data = zip_file.read(drawing_file)
                        chart_images = await self._extract_chart_images_from_drawing(
                            drawing_data, zip_file, sheet_name, sheet_index
                        )
                        embedded_images.extend(chart_images)
                    except Exception as e:
                        logger.error(f"Error processing drawing file {drawing_file}: {e}")
                        continue

            return embedded_images

        except Exception as e:
            logger.error(f"Error extracting embedded images: {e}")
            return []

    def _is_valid_image(self, image_data: bytes) -> bool:
        """Check if the data represents a valid image"""
        try:
            # Check common image file signatures
            image_signatures = [
                b"\xff\xd8\xff",  # JPEG
                b"\x89PNG\r\n\x1a\n",  # PNG
                b"GIF87a",  # GIF87a
                b"GIF89a",  # GIF89a
                b"BM",  # BMP
                b"RIFF",  # WEBP (starts with RIFF)
            ]

            for signature in image_signatures:
                if image_data.startswith(signature):
                    return True

            # Also try to open with PIL as a final check
            try:
                Image.open(BytesIO(image_data))
                return True
            except:
                return False

        except Exception:
            return False

    async def _extract_chart_images_from_drawing(
        self,
        drawing_data: bytes,
        zip_file: zipfile.ZipFile,
        sheet_name: str,
        sheet_index: int,
    ) -> List[Dict[str, Any]]:
        """Extract chart images from drawing XML"""
        chart_images = []

        try:
            # Parse the drawing XML
            root = ET.fromstring(drawing_data)

            # Look for chart references in the XML
            # This is a simplified approach - real implementation would need
            # to handle all the XML namespaces and relationships properly

            # For now, we'll look for any embedded images in the drawing
            for elem in root.iter():
                # Look for image references
                if "embed" in elem.attrib:
                    embed_id = elem.attrib["embed"]
                    # This would normally require parsing the relationships
                    # to map embed IDs to actual files

        except Exception as e:
            logger.error(f"Error parsing drawing XML: {e}")

        return chart_images

    async def _search_relevant_images(
        self, images_data: List[Dict[str, Any]], search_term: str, max_results: int
    ) -> List[ImageSearchResult]:
        """Search for relevant images/charts with improved ranking including text proximity"""

        if not images_data:
            return []

        # Extract text and image positions for proximity analysis
        logger.info(f"Extracting text and image positions for proximity analysis...")
        # We need the original Excel data for position analysis
        # For now, we'll use a simplified approach and enhance it later
        
        # Create tasks to process images in parallel
        semaphore = asyncio.Semaphore(3)
        tasks = []

        # Check if we should use optimized processing (check for global constant)
        use_optimized = getattr(self, 'enable_fast_processing', True)

        for img_data in images_data:
            if use_optimized:
                # Use optimized method with no API calls
                task = self._analyze_image_relevance_optimized(semaphore, img_data, search_term)
            else:
                # Use full method with API calls
                task = self._analyze_image_relevance_with_proximity(semaphore, img_data, search_term)
            tasks.append(task)

        # Execute analysis in parallel
        analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter and sort results with improved logic - only images with >75% confidence
        valid_results = []
        for i, result in enumerate(analysis_results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing image {i}: {result}")
                continue

            if result and result.confidence > IMAGE_CONFIDENCE_THRESHOLD:  # Increased threshold to 75% for better precision
                valid_results.append(result)

        # Sort by confidence (descending), then by image type (embedded first), then by page number
        valid_results.sort(
            key=lambda x: (
                -x.confidence,  # Higher confidence first
                (
                    0
                    if hasattr(x, "sheet_name")
                    and "embedded" in str(x.sheet_name).lower()
                    else 1
                ),  # Embedded images first
                x.page_number,  # Lower page numbers first as tiebreaker
            )
        )

        # Log the ranking for debugging
        logger.info(f"Excel Images ranked by relevance for '{search_term}' (confidence >{IMAGE_CONFIDENCE_THRESHOLD}, including proximity analysis):")
        for i, result in enumerate(valid_results[:max_results]):
            sheet_info = f" (Sheet: {result.sheet_name})" if result.sheet_name else ""
            logger.info(
                f"  {i+1}. Page {result.page_number}{sheet_info} - Confidence: {result.confidence:.3f}"
            )

        return valid_results[:max_results]

    async def _analyze_image_relevance(
        self, semaphore: asyncio.Semaphore, img_data: Dict[str, Any], search_term: str
    ) -> Optional[ImageSearchResult]:
        """Analyze chart/image relevance to search term"""

        async with semaphore:
            try:
                # Analyze image with Python tools
                python_analysis = self._analyze_image_with_python_tools(
                    img_data["image_base64"]
                )

                # Calculate relevance for Excel charts
                relevance_score = self._calculate_excel_relevance_score(
                    python_analysis, search_term, img_data
                )

                if relevance_score > 0.3:
                    # Create description for Excel chart
                    description = self._create_excel_description(
                        python_analysis, search_term, img_data
                    )

                    return ImageSearchResult(
                        page_number=img_data["page_number"],
                        sheet_name=img_data.get("sheet_name"),
                        image_base64=img_data["image_base64"],
                        description=description,
                        confidence=relevance_score,
                    )

                return None

            except Exception as e:
                logger.error(f"Error analyzing chart relevance: {e}")
                return None

    async def _analyze_image_relevance_with_proximity(
        self, semaphore: asyncio.Semaphore, img_data: Dict[str, Any], search_term: str
    ) -> Optional[ImageSearchResult]:
        """Analyze chart/image relevance to search term including text proximity"""

        async with semaphore:
            try:
                # Analyze image with Python tools
                python_analysis = self._analyze_image_with_python_tools(
                    img_data["image_base64"]
                )

                # Calculate base relevance for Excel charts
                base_relevance_score = self._calculate_excel_relevance_score(
                    python_analysis, search_term, img_data
                )

                # Calculate proximity bonus using advanced analysis when available
                proximity_bonus = self._calculate_advanced_proximity_bonus(
                    search_term, img_data
                )

                # Combine base score with proximity bonus
                final_relevance_score = min(base_relevance_score + proximity_bonus, 1.0)

                if final_relevance_score > 0.3:
                    # Create description for Excel chart
                    description = self._create_excel_description_with_proximity(
                        python_analysis, search_term, img_data, proximity_bonus
                    )

                    return ImageSearchResult(
                        page_number=img_data["page_number"],
                        sheet_name=img_data.get("sheet_name"),
                        image_base64=img_data["image_base64"],
                        description=description,
                        confidence=final_relevance_score,
                    )

                return None

            except Exception as e:
                logger.error(f"Error analyzing chart relevance with proximity: {e}")
                return None

    async def _analyze_image_relevance_optimized(
        self, semaphore: asyncio.Semaphore, img_data: Dict[str, Any], search_term: str
    ) -> Optional[ImageSearchResult]:
        """Optimized image analysis that avoids API calls for faster processing"""

        async with semaphore:
            try:
                # Analyze image with Python tools only (no API calls)
                python_analysis = self._analyze_image_with_python_tools(
                    img_data["image_base64"]
                )

                # Use optimized relevance calculation (no API calls)
                relevance_score = self._calculate_relevance_score_optimized(
                    python_analysis, search_term, img_data
                )

                # Add simple proximity bonus
                proximity_bonus = self._calculate_simplified_proximity_bonus(
                    search_term, img_data
                )

                # Combine scores
                final_relevance_score = min(relevance_score + proximity_bonus, 1.0)

                if final_relevance_score > IMAGE_CONFIDENCE_THRESHOLD:
                    # Create optimized description (no API calls)
                    description = self._create_optimized_description(
                        python_analysis, search_term, img_data
                    )

                    return ImageSearchResult(
                        page_number=img_data["page_number"],
                        sheet_name=img_data.get("sheet_name"),
                        image_base64=img_data["image_base64"],
                        description=description,
                        confidence=final_relevance_score,
                    )

                return None

            except Exception as e:
                logger.error(f"Error in optimized chart analysis: {e}")
                return None

    def _calculate_advanced_proximity_bonus(
        self, search_term: str, img_data: Dict[str, Any]
    ) -> float:
        """Calculate proximity bonus using advanced position analysis when available"""
        try:
            proximity_bonus = 0.0
            
            # Use detailed position analysis if available
            if hasattr(self, 'positions_data') and self.positions_data:
                proximity_bonus = self._calculate_text_image_proximity(
                    search_term, img_data, self.positions_data
                )
            
            # If no detailed position data, use simplified approach
            if proximity_bonus == 0.0:
                proximity_bonus = self._calculate_simplified_proximity_bonus(search_term, img_data)
            
            return proximity_bonus
            
        except Exception as e:
            logger.error(f"Error calculating advanced proximity bonus: {e}")
            return self._calculate_simplified_proximity_bonus(search_term, img_data)

    def _calculate_simplified_proximity_bonus(
        self, search_term: str, img_data: Dict[str, Any]
    ) -> float:
        """Calculate a simplified proximity bonus based on available data"""
        try:
            proximity_bonus = 0.0
            search_term_lower = search_term.lower()
            
            # Check if this is a specific product search
            is_specific_search = self._is_specific_product_search(search_term_lower)
            
            # Bonus based on sheet name relevance (text likely to be in same sheet)
            sheet_name = img_data.get("sheet_name", "").lower()
            if search_term_lower in sheet_name:
                proximity_bonus += 0.20 if is_specific_search else 0.15  # Higher bonus for specific searches
            
            # Check for partial matches in sheet name for multi-word searches
            if is_specific_search and len(search_term_lower.split()) > 1:
                search_words = search_term_lower.split()
                words_in_sheet = sum(1 for word in search_words if word in sheet_name)
                if words_in_sheet > 0:
                    proximity_bonus += 0.10 * (words_in_sheet / len(search_words))
            
            # Bonus for embedded images (more likely to be near text)
            image_type = img_data.get("image_type", "")
            if image_type == "embedded":
                proximity_bonus += 0.12 if is_specific_search else 0.10  # Higher bonus for specific searches
            
            # Bonus based on extraction method
            extraction_method = img_data.get("extraction_method", "")
            if "embedded" in extraction_method:
                proximity_bonus += 0.08
            
            # Bonus for images that appear earlier in the sheet (first page proximity)
            page_number = img_data.get("page_number", 1)
            if page_number == 1:
                proximity_bonus += 0.05  # First sheet often has relevant content
            
            # Additional bonuses for specific search terms that suggest proximity
            proximity_keywords = [
                "diagram", "illustration", "example", "figure", "image", 
                "picture", "logo", "chart", "graph", "table"
            ]
            
            if any(keyword in search_term_lower for keyword in proximity_keywords):
                proximity_bonus += 0.12  # These terms suggest text-image relationship
            
            # For specific product searches, reduce bonus if no strong indicators
            if is_specific_search and proximity_bonus < 0.15:
                proximity_bonus *= 0.7  # Reduce bonus for weak proximity indicators
            
            return min(proximity_bonus, 0.30)  # Increased cap for better precision
            
        except Exception as e:
            logger.error(f"Error calculating simplified proximity: {e}")
            return 0.0

    def _find_text_matches_in_sheet(
        self, search_term: str, sheet_name: str
    ) -> List[Dict[str, Any]]:
        """Find all text matches for search term in a specific sheet"""
        try:
            if not hasattr(self, 'positions_data') or not self.positions_data:
                return []
            
            sheet_text = self.positions_data["sheets_text_content"].get(sheet_name, [])
            search_term_lower = search_term.lower()
            
            matches = []
            for text_item in sheet_text:
                if search_term_lower in text_item["text"].lower():
                    matches.append({
                        "text": text_item["text"],
                        "x_position": text_item.get("x_position", 0),
                        "y_position": text_item.get("y_position", 0),
                        "row": text_item.get("row", 0),
                        "column": text_item.get("column", 0),
                        "match_score": self._calculate_text_match_score(
                            text_item["text"], search_term
                        )
                    })
            
            # Sort by match score (best matches first)
            matches.sort(key=lambda x: x["match_score"], reverse=True)
            return matches
            
        except Exception as e:
            logger.error(f"Error finding text matches: {e}")
            return []

    def _calculate_text_match_score(self, text: str, search_term: str) -> float:
        """Calculate how well a text matches the search term"""
        try:
            text_lower = text.lower()
            search_lower = search_term.lower()
            
            # For multi-word searches (like "coca cola"), require stricter matching
            if len(search_lower.split()) > 1:
                # Exact phrase match gets highest score
                if search_lower in text_lower:
                    return 1.0
                
                # Check if all words are present
                search_words = search_lower.split()
                text_words = text_lower.split()
                words_found = sum(1 for word in search_words if any(word in text_word for text_word in text_words))
                
                if words_found == len(search_words):
                    return 0.9  # All words found
                elif words_found >= len(search_words) * 0.7:
                    return 0.7  # Most words found
                else:
                    return 0.3  # Few words found
            
            # Single word searches
            # Exact match gets highest score
            if search_lower == text_lower:
                return 1.0
            
            # Word boundary match
            if f" {search_lower} " in f" {text_lower} ":
                return 0.9
            
            # Starts with search term
            if text_lower.startswith(search_lower):
                return 0.8
            
            # Contains search term
            if search_lower in text_lower:
                return 0.7
            
            # Partial word match
            words = text_lower.split()
            for word in words:
                if search_lower in word or word in search_lower:
                    return 0.6
            
            return 0.0
            
        except Exception:
            return 0.0

    async def analyze_excel_data(self, excel_url: str) -> Dict[str, Any]:
        """Analyze Excel file and return comprehensive data analysis"""
        try:
            response = requests.get(excel_url)
            response.raise_for_status()
            excel_data = response.content

            excel_file = pd.ExcelFile(BytesIO(excel_data))

            analysis = {
                "sheets": [],
                "total_sheets": len(excel_file.sheet_names),
                "summary": {},
            }

            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(BytesIO(excel_data), sheet_name=sheet_name)

                    sheet_analysis = {
                        "name": sheet_name,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "numeric_columns": len(
                            df.select_dtypes(include=[np.number]).columns
                        ),
                        "text_columns": len(
                            df.select_dtypes(include=["object"]).columns
                        ),
                        "has_data": not df.empty,
                        "column_names": df.columns.tolist(),
                    }

                    analysis["sheets"].append(sheet_analysis)

                except Exception as e:
                    logger.error(f"Error analyzing sheet {sheet_name}: {e}")
                    continue

            return analysis

        except Exception as e:
            return {"error": f"Error analyzing Excel file: {str(e)}"}

    async def _extract_text_and_image_positions(
        self, excel_data: bytes
    ) -> Dict[str, Any]:
        """Extract text content and image positions from Excel file"""
        try:
            positions_data = {
                "text_positions": {},
                "image_positions": {},
                "sheets_text_content": {}
            }
            
            # Extract text content from all sheets
            excel_file = pd.ExcelFile(BytesIO(excel_data))
            
            for sheet_index, sheet_name in enumerate(excel_file.sheet_names):
                try:
                    df = pd.read_excel(BytesIO(excel_data), sheet_name=sheet_name)
                    
                    if df.empty:
                        continue
                    
                    # Store text content with approximate positions
                    sheet_text_data = []
                    for row_idx, row in df.iterrows():
                        for col_idx, cell_value in enumerate(row):
                            if pd.notna(cell_value) and str(cell_value).strip():
                                sheet_text_data.append({
                                    "text": str(cell_value),
                                    "row": row_idx,
                                    "column": col_idx,
                                    "x_position": col_idx * 100,  # Approximate X position
                                    "y_position": row_idx * 20    # Approximate Y position
                                })
                    
                    positions_data["sheets_text_content"][sheet_name] = sheet_text_data
                    
                except Exception as e:
                    logger.error(f"Error extracting text positions from sheet {sheet_name}: {e}")
                    continue
            
            # Extract image positions (simplified approach)
            # In a full implementation, we would parse the drawing XML files
            # to get exact coordinates, but for now we'll use sheet-based approximation
            with zipfile.ZipFile(BytesIO(excel_data), "r") as zip_file:
                file_list = zip_file.namelist()
                media_files = [f for f in file_list if f.startswith("xl/media/")]
                
                for idx, media_file in enumerate(media_files):
                    try:
                        # Approximate position based on order and sheet
                        sheet_association = await self._get_image_sheet_association(zip_file, media_file)
                        sheet_name = sheet_association.get("sheet_name", "Sheet1")
                        
                        # Estimate position (this would be more accurate with XML parsing)
                        estimated_position = {
                            "x_position": (idx % 3) * 300 + 200,  # Distribute across columns
                            "y_position": (idx // 3) * 200 + 100, # Stack vertically
                            "sheet_name": sheet_name,
                            "media_file": media_file
                        }
                        
                        positions_data["image_positions"][media_file] = estimated_position
                        
                    except Exception as e:
                        logger.error(f"Error extracting position for {media_file}: {e}")
                        continue
            
            return positions_data
            
        except Exception as e:
            logger.error(f"Error extracting positions: {e}")
            return {"text_positions": {}, "image_positions": {}, "sheets_text_content": {}}

    def _calculate_text_image_proximity(
        self, 
        search_term: str,
        img_data: Dict[str, Any],
        positions_data: Dict[str, Any]
    ) -> float:
        """Calculate proximity bonus based on distance between search term and image"""
        try:
            proximity_bonus = 0.0
            search_term_lower = search_term.lower()
            
            sheet_name = img_data.get("sheet_name", "")
            media_file = img_data.get("media_file", "")
            
            # Get image position
            image_position = positions_data["image_positions"].get(media_file)
            if not image_position:
                return proximity_bonus
            
            img_x = image_position.get("x_position", 0)
            img_y = image_position.get("y_position", 0)
            
            # Get text content for the same sheet
            sheet_text = positions_data["sheets_text_content"].get(sheet_name, [])
            if not sheet_text:
                return proximity_bonus
            
            # Find text that matches search term
            matching_texts = []
            for text_item in sheet_text:
                if search_term_lower in text_item["text"].lower():
                    matching_texts.append(text_item)
            
            if not matching_texts:
                return proximity_bonus
            
            # Calculate proximity for each matching text
            min_distance = float('inf')
            closest_text = None
            
            for text_item in matching_texts:
                text_x = text_item.get("x_position", 0)
                text_y = text_item.get("y_position", 0)
                
                # Calculate distance (emphasizing X-axis proximity as requested)
                x_distance = abs(img_x - text_x)
                y_distance = abs(img_y - text_y)
                
                # Weight X-axis distance more heavily (factor of 2)
                weighted_distance = (x_distance * 2) + y_distance
                
                if weighted_distance < min_distance:
                    min_distance = weighted_distance
                    closest_text = text_item
            
            # Convert distance to proximity bonus
            if min_distance < float('inf'):
                # Maximum bonus for very close proximity (same row/column area)
                if min_distance <= 150:  # Very close
                    proximity_bonus = 0.25
                elif min_distance <= 300:  # Close
                    proximity_bonus = 0.20
                elif min_distance <= 500:  # Moderately close
                    proximity_bonus = 0.15
                elif min_distance <= 800:  # Somewhat close
                    proximity_bonus = 0.10
                elif min_distance <= 1200:  # Same general area
                    proximity_bonus = 0.05
                
                # Additional bonus for same row (Y-axis alignment)
                if closest_text and abs(img_y - closest_text.get("y_position", 0)) <= 30:
                    proximity_bonus += 0.10  # Bonus for same horizontal line
                
                # Additional bonus for same column area (X-axis alignment)
                if closest_text and abs(img_x - closest_text.get("x_position", 0)) <= 120:
                    proximity_bonus += 0.08  # Bonus for same column area
            
            return min(proximity_bonus, 0.35)  # Cap the proximity bonus
            
        except Exception as e:
            logger.error(f"Error calculating proximity: {e}")
            return 0.0

    def _calculate_excel_relevance_score(
        self, analysis: Dict[str, Any], search_term: str, img_data: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for Excel charts and images with improved algorithm"""
        try:
            # Start with base score from parent class (improved algorithm)
            base_score = self._calculate_relevance_score(analysis, search_term)

            # Excel-specific bonuses
            chart_type = img_data.get("chart_type", "")
            image_type = img_data.get("image_type", "")
            search_term_lower = search_term.lower()

            # Check for specific product/brand searches that need higher precision
            is_specific_search = self._is_specific_product_search(search_term_lower)
            
            # For specific searches, require stronger evidence
            if is_specific_search:
                # Check if the media file name contains the search term
                media_file = img_data.get("media_file", "").lower()
                file_name_match = search_term_lower in media_file or any(
                    word in media_file for word in search_term_lower.split()
                )
                
                # Check sheet name for exact matches
                sheet_name = img_data.get("sheet_name", "").lower()
                sheet_name_match = search_term_lower in sheet_name or any(
                    word in sheet_name for word in search_term_lower.split()
                )
                
                # For specific searches, penalize if no direct evidence
                if not file_name_match and not sheet_name_match:
                    base_score *= 0.4  # Significant penalty for lack of direct evidence
                elif file_name_match:
                    base_score += 0.30  # Strong bonus for file name match
                elif sheet_name_match:
                    base_score += 0.20  # Good bonus for sheet name match

            # Different scoring for embedded images vs generated charts
            if image_type == "embedded":
                # For embedded images, add bonus for being actual content
                base_score += 0.15  # Embedded images are often more relevant

                # Bonus for specific search terms
                if search_term_lower in [
                    "image",
                    "picture",
                    "photo",
                    "product",
                    "logo",
                    "illustration",
                ]:
                    base_score += 0.25
                elif search_term_lower in ["content", "visual", "graphic"]:
                    base_score += 0.20

                # Media file name relevance (enhanced for specific searches)
                media_file = img_data.get("media_file", "").lower()
                if search_term_lower in media_file:
                    base_score += 0.25 if is_specific_search else 0.15

            else:
                # For generated charts, add chart-specific bonuses
                if search_term_lower in [
                    "chart",
                    "graph",
                    "visualization",
                    "data",
                    "analysis",
                ]:
                    base_score += 0.30
                elif search_term_lower in ["report", "summary", "statistics"]:
                    base_score += 0.25

                # Specific chart type bonuses
                chart_bonus = 0.0
                if "bar" in search_term_lower and chart_type == "bar":
                    chart_bonus = 0.20
                elif "line" in search_term_lower and chart_type == "line":
                    chart_bonus = 0.20
                elif "scatter" in search_term_lower and chart_type == "scatter":
                    chart_bonus = 0.20
                elif "heatmap" in search_term_lower and chart_type == "heatmap":
                    chart_bonus = 0.20
                elif "correlation" in search_term_lower and chart_type == "heatmap":
                    chart_bonus = 0.25

                base_score += chart_bonus

                # Generated charts generally have good quality
                if analysis.get("quality_score", 0.0) > 0.7:
                    base_score += 0.10

                # For specific product searches, charts are less likely to be relevant
                if is_specific_search and not any(word in search_term_lower for word in ["chart", "graph", "data"]):
                    base_score *= 0.6  # Reduce score for charts when searching for specific products

            # Sheet name relevance (applies to both types)
            sheet_name = img_data.get("sheet_name", "").lower()
            if search_term_lower in sheet_name:
                base_score += 0.20 if is_specific_search else 0.15

            # Bonus for sheets with meaningful names
            meaningful_sheet_names = [
                "data",
                "analysis",
                "report",
                "summary",
                "chart",
                "graph",
            ]
            if any(name in sheet_name for name in meaningful_sheet_names):
                base_score += 0.05

            # Ensure score is between 0 and 1
            return min(max(base_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating Excel relevance: {e}")
            return 0.0

    def _is_specific_product_search(self, search_term_lower: str) -> bool:
        """Determine if this is a specific product/brand search that needs higher precision"""
        # Common indicators of specific searches
        specific_indicators = [
            # Brands/Products
            "coca cola", "cocacola", "pepsi", "sprite", "fanta",
            "nike", "adidas", "apple", "samsung", "iphone",
            "cerveza", "beer", "wine", "vino",
            # Specific items
            "laptop", "computadora", "telefono", "celular",
            "camiseta", "pantalon", "zapatos",
            # Food items
            "pizza", "hamburguesa", "sandwich", "cafe", "coffee"
        ]
        
        # Check if it's a multi-word specific term
        if len(search_term_lower.split()) > 1:
            return True
            
        # Check against known specific terms
        return any(indicator in search_term_lower for indicator in specific_indicators)

    def _create_excel_description(
        self, analysis: Dict[str, Any], search_term: str, img_data: Dict[str, Any]
    ) -> str:
        """Create description for Excel chart or embedded image"""
        try:
            description_parts = []

            # Different descriptions for different image types
            image_type = img_data.get("image_type", "")
            chart_type = img_data.get("chart_type", "")
            sheet_name = img_data.get("sheet_name", "Unknown")

            if image_type == "embedded":
                # Description for embedded images
                description_parts.append(
                    f"Embedded image from Excel sheet '{sheet_name}'"
                )

                # Media file information
                media_file = img_data.get("media_file", "")
                if media_file:
                    file_name = media_file.split("/")[-1]
                    description_parts.append(f"(file: {file_name})")

            else:
                # Description for generated charts
                description_parts.append(
                    f"{chart_type.title()} chart from Excel sheet '{sheet_name}'"
                )

            # Technical details
            dimensions = analysis.get("dimensions", (0, 0))
            if dimensions[0] > 0 and dimensions[1] > 0:
                description_parts.append(f"({dimensions[0]}x{dimensions[1]} pixels)")

            # Quality assessment
            quality_score = analysis.get("quality_score", 0.0)
            if quality_score > 0.8:
                description_parts.append("with excellent visual quality")
            elif quality_score > 0.6:
                description_parts.append("with good quality")
            elif quality_score > 0.3:
                description_parts.append("with standard quality")

            # Content analysis
            objects_detected = analysis.get("objects_detected", 0)
            if objects_detected > 0:
                description_parts.append(
                    f"containing {objects_detected} visual elements"
                )

            # Text detection
            if analysis.get("text_detected", False):
                description_parts.append("with text content")

            # Colors
            dominant_colors = analysis.get("dominant_colors", [])
            if len(dominant_colors) > 0:
                color_description = self._describe_colors(dominant_colors)
                description_parts.append(f"with colors: {color_description}")

            # Brightness and contrast info
            brightness = analysis.get("brightness", 0)
            contrast = analysis.get("contrast", 0)
            if brightness > 0 and contrast > 0:
                if brightness > 200:
                    description_parts.append("bright appearance")
                elif brightness < 50:
                    description_parts.append("dark appearance")

                if contrast > 50:
                    description_parts.append("high contrast")

            # Relevance to search term
            description_parts.append(f"Related to search term: {search_term}")

            return ". ".join(description_parts).capitalize() + "."

        except Exception as e:
            image_type = img_data.get("image_type", "chart")
            sheet_name = img_data.get("sheet_name", "Unknown")
            return f"Excel {image_type} from sheet '{sheet_name}'. Related to {search_term}."

    def _create_excel_description_with_proximity(
        self, 
        analysis: Dict[str, Any], 
        search_term: str, 
        img_data: Dict[str, Any],
        proximity_bonus: float
    ) -> str:
        """Create description for Excel chart or embedded image including proximity info"""
        try:
            # Get base description
            base_description = self._create_excel_description(analysis, search_term, img_data)
            
            # Add proximity information if significant
            if proximity_bonus > 0.1:
                proximity_info = ""
                if proximity_bonus > 0.2:
                    proximity_info = " This image appears to be closely positioned relative to related text content."
                elif proximity_bonus > 0.15:
                    proximity_info = " This image appears to be near relevant text content."
                elif proximity_bonus > 0.1:
                    proximity_info = " This image may be positioned near related text."
                
                # Insert proximity info before the final period
                if base_description.endswith("."):
                    base_description = base_description[:-1] + proximity_info + "."
                else:
                    base_description += proximity_info
            
            return base_description
            
        except Exception as e:
            logger.error(f"Error creating description with proximity: {e}")
            return self._create_excel_description(analysis, search_term, img_data)

    def _describe_colors(self, colors: List[List[int]]) -> str:
        """Create a text description of dominant colors"""
        try:
            color_names = []
            for color in colors[:3]:  # Limit to top 3 colors
                if len(color) >= 3:
                    r, g, b = color[0], color[1], color[2]
                    color_name = self._get_color_name(r, g, b)
                    color_names.append(color_name)

            return ", ".join(color_names) if color_names else "various colors"

        except Exception:
            return "various colors"

    def _get_color_name(self, r: int, g: int, b: int) -> str:
        """Get approximate color name from RGB values"""
        try:
            # Simple color classification
            if r > 200 and g > 200 and b > 200:
                return "white"
            elif r < 50 and g < 50 and b < 50:
                return "black"
            elif r > 150 and g < 100 and b < 100:
                return "red"
            elif r < 100 and g > 150 and b < 100:
                return "green"
            elif r < 100 and g < 100 and b > 150:
                return "blue"
            elif r > 150 and g > 150 and b < 100:
                return "yellow"
            elif r > 150 and g < 100 and b > 150:
                return "purple"
            elif r < 100 and g > 150 and b > 150:
                return "cyan"
            elif r > 150 and g > 100 and b < 100:
                return "orange"
            elif r > 100 and g > 100 and b > 100:
                return "gray"
            else:
                return "mixed"

        except Exception:
            return "unknown"

    async def smart_search_with_text_analysis_excel(
        self, excel_url: str, search_phrase: str, max_results: int = 10
    ) -> ExcelSmartSearchResult:
        """
        Smart search in Excel that:
        1. Processes the search term (can be a phrase) using regex
        2. Extracts all text from all Excel sheets
        3. Searches for keyword matches in the text
        4. Returns related images based on matches

        Args:
            excel_url: URL of the Excel file
            search_phrase: Search term or phrase
            max_results: Maximum number of images to return

        Returns:
            ExcelSmartSearchResult with text analysis and related images
        """
        try:
            logger.info(f"🔍 Starting smart search in Excel for: '{search_phrase}'")
            
            # 1. Process the search term with regex to extract keywords
            keywords = self._extract_keywords_from_phrase_excel(search_phrase)
            logger.info(f"📝 Extracted keywords: {keywords}")
            
            # 2. Extract text from the Excel file
            extracted_text, extraction_method, sheets_analyzed = await self._extract_all_text_from_excel_comprehensive(excel_url)
            logger.info(f"📄 Extracted text from {len(sheets_analyzed)} sheets: {len(extracted_text)} characters")
            
            # 3. Search for matches in the text using regex
            text_matches = self._find_text_matches_with_regex_excel(extracted_text, keywords, search_phrase, sheets_analyzed)
            logger.info(f"🎯 Found {len(text_matches)} text matches")
            
            # 4. Find related images based on matches
            related_images = await self._find_excel_images_for_matches(
                excel_url, text_matches, search_phrase, max_results
            )
            logger.info(f"🖼️ Found {len(related_images)} related images")
            
            # 5. Construct final result
            result = ExcelSmartSearchResult(
                search_term=search_phrase,
                processed_keywords=keywords,
                text_matches=text_matches,
                related_images=related_images,
                total_matches=len(text_matches),
                extraction_method=extraction_method,
                sheets_analyzed=sheets_analyzed
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in smart Excel search: {str(e)}")
            # Return empty result in case of error
            return ExcelSmartSearchResult(
                search_term=search_phrase,
                processed_keywords=[],
                text_matches=[],
                related_images=[],
                total_matches=0,
                extraction_method="error",
                sheets_analyzed=[]
            )

    def _extract_keywords_from_phrase_excel(self, search_phrase: str) -> List[str]:
        """
        Extracts keywords from a phrase using regex for Excel
        """
        try:
            # Clean and normalize the phrase
            cleaned_phrase = search_phrase.lower().strip()
            
            # Common stopwords in Spanish
            stopwords = {
                'el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'y', 'a', 'que', 'es', 
                'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para',
                'del', 'al', 'me', 'mi', 'tu', 'si', 'o', 'pero', 'más', 'como', 'esta',
                'este', 'estos', 'estas', 'todo', 'todos', 'toda', 'todas'
            }
            
            # Extract words using regex (minimum 2 characters)
            words = re.findall(r'\b[a-záéíóúñ0-9]{2,}\b', cleaned_phrase, re.IGNORECASE)
            
            # Filter stopwords and duplicates
            keywords = []
            seen = set()
            
            for word in words:
                word_lower = word.lower()
                if word_lower not in stopwords and word_lower not in seen:
                    keywords.append(word)
                    seen.add(word_lower)
            
            # If no valid words were found, use the full phrase
            if not keywords:
                keywords = [search_phrase.strip()]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords from Excel: {e}")
            return [search_phrase.strip()]

    async def _extract_all_text_from_excel_comprehensive(self, excel_url: str) -> Tuple[str, str, List[str]]:
        """
        Extracts all text from an Excel file comprehensively
        """
        try:
            # Download Excel file
            response = requests.get(excel_url)
            response.raise_for_status()
            excel_data = response.content
            
            # Read all sheets
            excel_file = pd.ExcelFile(BytesIO(excel_data))
            all_text = ""
            sheets_analyzed = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(BytesIO(excel_data), sheet_name=sheet_name)
                    sheets_analyzed.append(sheet_name)
                    
                    if not df.empty:
                        all_text += f"\n[SHEET: {sheet_name}]\n"
                        
                        # Extract column names
                        column_text = " ".join([str(col) for col in df.columns if pd.notna(col)])
                        all_text += f"[COLUMNS: {column_text}]\n"
                        
                        # Extract text from all cells with position reference
                        for row_idx, row in df.iterrows():
                            row_text = ""
                            for col_name, value in row.items():
                                if pd.notna(value) and str(value).strip():
                                    cell_ref = f"{col_name}_{row_idx + 1}"
                                    row_text += f"[{cell_ref}:{str(value)}] "
                            
                            if row_text.strip():
                                all_text += f"ROW_{row_idx + 1}: {row_text}\n"
                        
                        all_text += "\n"
                
                except Exception as e:
                    logger.error(f"Error processing sheet {sheet_name}: {e}")
                    continue
            
            return all_text, "excel_comprehensive_extraction", sheets_analyzed
            
        except Exception as e:
            logger.error(f"Error extracting text from Excel: {e}")
            return "", "error", []

    def _find_text_matches_with_regex_excel(
        self, text: str, keywords: List[str], original_phrase: str, sheets_analyzed: List[str]
    ) -> List[ExcelTextSearchMatch]:
        """
        Searches for matches in Excel text using regex for keywords
        """
        try:
            matches = []
            text_lower = text.lower()
            
            # Split text by sheets
            sheets = text.split('[SHEET:')
            
            for sheet_idx, sheet_content in enumerate(sheets):
                if sheet_idx == 0:
                    continue  # Skip the part before any sheet
                
                # Extract sheet name
                sheet_lines = sheet_content.split('\n')
                if not sheet_lines:
                    continue
                
                sheet_name = sheet_lines[0].strip().rstrip(']')
                sheet_text = '\n'.join(sheet_lines[1:])
                sheet_text_lower = sheet_text.lower()
                
                # Search for full phrase first
                phrase_pattern = re.escape(original_phrase.lower())
                for match in re.finditer(phrase_pattern, sheet_text_lower):
                    # Extract context and cell reference
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(sheet_text), match.end() + 100)
                    context = sheet_text[context_start:context_end].strip()
                    
                    # Attempt to extract cell reference from context
                    cell_ref, row_num, col_name = self._extract_cell_reference_from_context(context)
                    
                    matches.append(ExcelTextSearchMatch(
                        sheet_name=sheet_name,
                        cell_reference=cell_ref,
                        matched_text=original_phrase,
                        match_context=context,
                        confidence=1.0,  # Exact phrase match
                        row_number=row_num,
                        column_name=col_name
                    ))
                
                # Search for individual keywords
                for keyword in keywords:
                    # Create regex pattern to search for the full keyword
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    
                    for match in re.finditer(pattern, sheet_text_lower):
                        # Extract context around the match
                        context_start = max(0, match.start() - 100)
                        context_end = min(len(sheet_text), match.end() + 100)
                        context = sheet_text[context_start:context_end].strip()
                        
                        # Attempt to extract cell reference from context
                        cell_ref, row_num, col_name = self._extract_cell_reference_from_context(context)
                        
                        # Calculate confidence based on keyword length
                        confidence = min(0.8, 0.4 + (len(keyword) * 0.1))
                        
                        matches.append(ExcelTextSearchMatch(
                            sheet_name=sheet_name,
                            cell_reference=cell_ref,
                            matched_text=keyword,
                            match_context=context,
                            confidence=confidence,
                            row_number=row_num,
                            column_name=col_name
                        ))
            
            # Sort by confidence and remove duplicates
            unique_matches = []
            seen_contexts = set()
            
            for match in sorted(matches, key=lambda x: x.confidence, reverse=True):
                context_key = (match.sheet_name, match.matched_text.lower(), match.match_context[:50])
                if context_key not in seen_contexts:
                    unique_matches.append(match)
                    seen_contexts.add(context_key)
            
            return unique_matches[:MAX_IMAGES_TO_SHOW]  # Limit to max images
            
        except Exception as e:
            logger.error(f"Error searching for text matches in Excel: {e}")
            return []

    def _extract_cell_reference_from_context(self, context: str) -> Tuple[str, int, str]:
        """
        Extracts the cell reference from the context
        """
        try:
            # Search for patterns like [COLUMN_ROW:value] or ROW_X:
            cell_pattern = r'\[([^:]+)_(\d+):[^\]]+\]'
            cell_match = re.search(cell_pattern, context)
            
            if cell_match:
                column_name = cell_match.group(1)
                row_number = int(cell_match.group(2))
                cell_ref = f"{column_name}{row_number}"
                return cell_ref, row_number, column_name
            
            # Search for ROW_X: pattern
            row_pattern = r'ROW_(\d+):'
            row_match = re.search(row_pattern, context)
            
            if row_match:
                row_number = int(row_match.group(1))
                return f"ROW_{row_number}", row_number, "MULTIPLE"
            
            # If no specific pattern found
            return "UNKNOWN", 0, "UNKNOWN"
            
        except Exception as e:
            logger.error(f"Error extracting cell reference: {e}")
            return "ERROR", 0, "ERROR"

    async def _find_excel_images_for_matches(
        self, excel_url: str, text_matches: List[ExcelTextSearchMatch], 
        search_phrase: str, max_results: int
    ) -> List[ImageSearchResult]:
        """
        Searches for images in Excel based on text matches
        """
        try:
            # Get sheets with matches
            match_sheets = list(set(match.sheet_name for match in text_matches))
            logger.info(f"🔍 Searching for images in sheets with matches: {match_sheets}")
            
            # Use existing search method with term
            excel_images = await self.search_images_in_excel(excel_url, search_phrase, max_results)
            
            # Prioritize images if there are text matches
            if text_matches:
                for img in excel_images:
                    # Increase confidence for images when there are text matches
                    img.confidence = min(1.0, img.confidence + 0.3)
                    
                    # Add match information
                    img.description += f" [File with {len(text_matches)} matches in {len(match_sheets)} sheets]"
                    
                    # If the image is in a sheet with matches, increase confidence further
                    for match in text_matches:
                        if match.sheet_name in img.description or str(img.page_number) in [match.sheet_name]:
                            img.confidence = min(1.0, img.confidence + 0.2)
                            img.description += f" [Match in sheet: {match.sheet_name}]"
                            break
            
            return excel_images
            
        except Exception as e:
            logger.error(f"Error searching for images in Excel: {e}")
            return []

    # Demo function for Excel
    async def demo_smart_search_excel(self, excel_url: str):
        """
        Demo function to show how to use smart search in Excel
        """
        logger.info("🚀 DEMO: Smart Excel Search with Text Analysis and Regex")
        logger.info("=" * 60)
        
        # Example search phrases for Excel
        search_phrases = [
            "monthly sales",
            "product inventory stock",
            "employee resources",
            "financial budget expenses",
            "client contacts"
        ]
        
        for phrase in search_phrases:
            logger.info(f"\n�� Searching in Excel: '{phrase}'")
            logger.info("-" * 40)
            
            result = await self.smart_search_with_text_analysis_excel(
                excel_url=excel_url,
                search_phrase=phrase,
                max_results=5
            )
            
            logger.info(f"📝 Extracted keywords: {result.processed_keywords}")
            logger.info(f"🎯 Found {result.total_matches} text matches")
            logger.info(f"📄 Extraction method: {result.extraction_method}")
            logger.info(f"📊 Analyzed sheets: {result.sheets_analyzed}")
            logger.info(f"🖼️ Related images: {len(result.related_images)}")
            
            # Display some text matches
            if result.text_matches:
                logger.info("\n📋 First text matches:")
                for i, match in enumerate(result.text_matches[:3]):
                    logger.info(f"  {i+1}. Sheet '{match.sheet_name}', Cell {match.cell_reference}: '{match.matched_text}' "
                          f"(confidence: {match.confidence:.2f})")
                    logger.info(f"     Context: {match.match_context[:100]}...")
            
            # Display image information
            if result.related_images:
                logger.info("\n🖼️ Found images:")
                for i, img in enumerate(result.related_images[:3]):
                    logger.info(f"  {i+1}. {img.description[:100]}... "
                          f"(confidence: {img.confidence:.2f})")
            
            logger.info("\n" + "="*60)
        
        return "Excel demo completed"
