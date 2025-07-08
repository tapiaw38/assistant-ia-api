# Assistant IA API

This project is a Python-based API designed to provide intelligent assistant functionalities. It leverages modern AI technologies to deliver efficient and scalable solutions.

## Features

- Natural Language Processing (NLP) capabilities.
- RESTful API design for easy integration.
- Scalable and modular architecture.
- Support for multiple AI models.
- **NEW: PDF Text and Image Extraction** - Advanced PDF processing with OCR support
  - Automatic text detection and extraction from PDFs
  - OCR processing for image-based PDFs (scanned documents)
  - Smart image recognition and contextual analysis
  - Related image detection based on extracted text
  - Multi-language support (Spanish/English) for OCR

## New PDF Processing Capabilities

### Automatic Content Detection

- Detects whether PDF contains native text or requires OCR processing
- Intelligently switches between text extraction methods
- Provides confidence scores for extraction quality

### OCR (Optical Character Recognition)

- Processes embedded images within PDFs
- Converts full pages to images when necessary
- Uses Tesseract with advanced preprocessing for optimal results
- Supports Spanish and English text recognition

### Smart Image Analysis

- Identifies images related to extracted text content
- Generates contextual descriptions for images
- Calculates relevance scores between text and images
- Filters images based on search terms

### Usage Example

```python
from src.core.platform.fileprocessor.pdf_processor import FileImageProcessor

# Extract text and related images from PDF
result = await processor.extract_text_and_images(
    pdf_url="https://example.com/document.pdf",
    search_term="quilted"  # Optional filter
)

# Access extracted content
text = result["extracted_text"]["full_text"]
images = result["related_images"]["images"]
confidence = result["extracted_text"]["avg_confidence"]
```

For detailed documentation, see [PDF Text Extraction Guide](docs/pdf_text_extraction.md)

## Requirements

- Python 3.8 or higher installed.
- Docker and Docker Compose installed on your system.
- Make sure `make` is available on your system to use the provided Makefile for managing the project.
