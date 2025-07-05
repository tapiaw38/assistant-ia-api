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
from src.core.platform.fileprocessor.processor import (
    BaseFileProcessor,
    ImageSearchResult,
)


class ExcelImageProcessor(BaseFileProcessor):
    """Excel file processor that can extract and analyze embedded images and charts/graphs"""

    def __init__(self, config_service):
        super().__init__(config_service)

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
                    print(f"Error processing sheet {sheet_name}: {e}")
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
                        print(f"Error extracting media file {media_file}: {e}")
                        continue

            return embedded_images

        except Exception as e:
            print(f"Error extracting embedded images: {e}")
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
            print(f"Error determining sheet association: {e}")
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
                    print(
                        f"Error creating {chart_type} chart for sheet {sheet_name}: {e}"
                    )
                    continue

            return charts_data

        except Exception as e:
            print(f"Error generating charts from sheet {sheet_name}: {e}")
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
            print(f"Error creating bar chart: {e}")
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
            print(f"Error creating line chart: {e}")
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
            print(f"Error creating scatter chart: {e}")
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
            print(f"Error creating heatmap: {e}")
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
                        print(f"Error extracting media file {media_file}: {e}")
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
                        print(f"Error processing drawing file {drawing_file}: {e}")
                        continue

            return embedded_images

        except Exception as e:
            print(f"Error extracting embedded images: {e}")
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
            print(f"Error parsing drawing XML: {e}")

        return chart_images

    async def _search_relevant_images(
        self, images_data: List[Dict[str, Any]], search_term: str, max_results: int
    ) -> List[ImageSearchResult]:
        """Search for relevant images/charts with improved ranking"""

        if not images_data:
            return []

        # Create tasks to process images in parallel
        semaphore = asyncio.Semaphore(3)
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

            if result and result.confidence > 0.25:  # Lower threshold for more results
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
        print(f"Excel Images ranked by relevance for '{search_term}':")
        for i, result in enumerate(valid_results[:max_results]):
            sheet_info = f" (Sheet: {result.sheet_name})" if result.sheet_name else ""
            print(
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
                print(f"Error analyzing chart relevance: {e}")
                return None

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

                # Media file name relevance
                media_file = img_data.get("media_file", "").lower()
                if search_term_lower in media_file:
                    base_score += 0.15

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

            # Sheet name relevance (applies to both types)
            sheet_name = img_data.get("sheet_name", "").lower()
            if search_term_lower in sheet_name:
                base_score += 0.15

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
            print(f"Error calculating Excel relevance: {e}")
            return 0.0

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
                    print(f"Error analyzing sheet {sheet_name}: {e}")
                    continue

            return analysis

        except Exception as e:
            return {"error": f"Error analyzing Excel file: {str(e)}"}
