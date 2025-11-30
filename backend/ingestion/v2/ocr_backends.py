"""
OCR Backend Abstraction for Document Understanding.

Provides pluggable backends for extracting structured pages from PDFs:
- PyMuPDFBackend: Fast, local, text-layer PDFs
- GeminiVisionBackend: API-based, handles complex layouts and scanned pages
- OpenAIVisionBackend: API-based alternative
- ClaudeVisionBackend: API-based alternative

The backends normalize output to StructuredPage format.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .models import (
    BlockType,
    BoundingBox,
    PageBlock,
    StructuredPage,
)

logger = logging.getLogger("backend.ingestion.v2.ocr")


# =============================================================================
# Base OCR Backend
# =============================================================================

class OCRBackend(ABC):
    """Abstract base class for OCR backends."""
    
    name: str = "base"
    
    @abstractmethod
    def extract_pages(
        self,
        file_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Iterator[StructuredPage]:
        """
        Extract structured pages from a PDF.
        
        Args:
            file_path: Path to the PDF file
            page_range: Optional (start, end) page range (1-indexed, inclusive)
            
        Yields:
            StructuredPage objects
        """
        pass
    
    def extract_all_pages(
        self,
        file_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[StructuredPage]:
        """Extract all pages as a list."""
        return list(self.extract_pages(file_path, page_range))
    
    def is_page_scanned(self, page_text: str, page_char_count: int) -> bool:
        """Heuristic to detect if a page is scanned (image-based)."""
        # If very little text extracted relative to page size, likely scanned
        if page_char_count < 50:
            return True
        # If text has many replacement characters or garbage
        garbage_ratio = sum(1 for c in page_text if ord(c) > 0xFFFF or c == '\ufffd') / max(len(page_text), 1)
        return garbage_ratio > 0.1


# =============================================================================
# PyMuPDF Backend (Local, Fast)
# =============================================================================

class PyMuPDFBackend(OCRBackend):
    """
    Local PDF text extraction using PyMuPDF (fitz).
    
    Best for:
    - PDFs with embedded text layer
    - Fast processing
    - No API costs
    
    Limitations:
    - Cannot handle scanned pages
    - Limited layout understanding
    """
    
    name = "pymupdf"
    
    def __init__(self, detect_structure: bool = True):
        self.detect_structure = detect_structure
    
    def extract_pages(
        self,
        file_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Iterator[StructuredPage]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
        
        doc = fitz.open(file_path)
        
        start_page = (page_range[0] - 1) if page_range else 0
        end_page = page_range[1] if page_range else len(doc)
        
        for page_idx in range(start_page, min(end_page, len(doc))):
            page = doc[page_idx]
            page_number = page_idx + 1
            
            # Get page dimensions
            rect = page.rect
            width, height = rect.width, rect.height
            
            # Extract text blocks with positions
            blocks = []
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            
            order_index = 0
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    bbox = BoundingBox(
                        x0=block["bbox"][0],
                        y0=block["bbox"][1],
                        x1=block["bbox"][2],
                        y1=block["bbox"][3],
                    )
                    
                    # Extract text from lines
                    text_parts = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_parts.append(span.get("text", ""))
                    
                    text = " ".join(text_parts).strip()
                    if not text:
                        continue
                    
                    # Detect block type from formatting
                    block_type = self._detect_block_type(text, block, text_dict)
                    
                    # Check for equations (LaTeX patterns)
                    latex = None
                    if self._looks_like_equation(text):
                        block_type = BlockType.EQUATION
                        latex = text
                    
                    blocks.append(PageBlock(
                        block_type=block_type,
                        text=text,
                        bbox=bbox,
                        latex=latex,
                        order_index=order_index,
                        confidence=1.0,
                    ))
                    order_index += 1
                
                elif block.get("type") == 1:  # Image block
                    bbox = BoundingBox(
                        x0=block["bbox"][0],
                        y0=block["bbox"][1],
                        x1=block["bbox"][2],
                        y1=block["bbox"][3],
                    )
                    blocks.append(PageBlock(
                        block_type=BlockType.FIGURE,
                        text="[Figure]",
                        bbox=bbox,
                        order_index=order_index,
                    ))
                    order_index += 1
            
            # Get raw text
            raw_text = page.get_text()
            
            # Check if scanned
            is_scanned = self.is_page_scanned(raw_text, len(raw_text))
            
            yield StructuredPage(
                page_number=page_number,
                blocks=blocks,
                width=width,
                height=height,
                raw_text=raw_text,
                ocr_backend=self.name,
                is_scanned=is_scanned,
            )
        
        doc.close()
    
    def _detect_block_type(
        self,
        text: str,
        block: Dict,
        page_dict: Dict,
    ) -> BlockType:
        """Detect block type from text and formatting."""
        text_lower = text.lower().strip()
        
        # Check for headings by font size
        avg_font_size = self._get_avg_font_size(block)
        page_avg_font = self._get_page_avg_font_size(page_dict)
        
        if avg_font_size and page_avg_font:
            size_ratio = avg_font_size / page_avg_font
            if size_ratio > 1.5:
                return BlockType.HEADING_1
            elif size_ratio > 1.3:
                return BlockType.HEADING_2
            elif size_ratio > 1.15:
                return BlockType.HEADING_3
        
        # Pattern-based detection
        # Chapter/Section headings
        if re.match(r'^(chapter|section|\d+\.)\s', text_lower):
            if re.match(r'^chapter\s+\d+', text_lower):
                return BlockType.HEADING_1
            elif re.match(r'^\d+\.\d+\s', text_lower):
                return BlockType.HEADING_3
            elif re.match(r'^\d+\.\s', text_lower):
                return BlockType.HEADING_2
        
        # Definition patterns
        if any(p in text_lower for p in ["is defined as", "definition:", "we define"]):
            return BlockType.PARAGRAPH  # Still paragraph, but will be tagged as definition
        
        # List items
        if re.match(r'^[\•\-\*\d+\.]\s', text):
            return BlockType.LIST_ITEM
        
        # Footnotes (usually at bottom, smaller font)
        if text_lower.startswith("note:") or re.match(r'^\d+\s', text) and len(text) < 200:
            return BlockType.FOOTNOTE
        
        return BlockType.PARAGRAPH
    
    def _get_avg_font_size(self, block: Dict) -> Optional[float]:
        """Get average font size in a block."""
        sizes = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "size" in span:
                    sizes.append(span["size"])
        return sum(sizes) / len(sizes) if sizes else None
    
    def _get_page_avg_font_size(self, page_dict: Dict) -> Optional[float]:
        """Get average font size across the page."""
        sizes = []
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if "size" in span:
                            sizes.append(span["size"])
        return sum(sizes) / len(sizes) if sizes else None
    
    def _looks_like_equation(self, text: str) -> bool:
        """Check if text looks like a mathematical equation."""
        # LaTeX patterns
        if re.search(r'\\[a-zA-Z]+', text):  # LaTeX commands
            return True
        if re.search(r'\$[^$]+\$', text):  # Inline math
            return True
        
        # Math symbols density
        math_chars = set('=∫∑∏∂∇√∞±≈≠≤≥∈∉⊂⊃∀∃αβγδεζηθλμνξπρστφχψω')
        math_density = sum(1 for c in text if c in math_chars) / max(len(text), 1)
        if math_density > 0.1:
            return True
        
        # Simple equation pattern (e.g., "F = ma")
        if re.match(r'^[A-Za-z_]\s*=\s*[A-Za-z0-9_\s\+\-\*/\^]+$', text.strip()):
            return True
        
        return False


# =============================================================================
# Gemini Vision Backend (API-based)
# =============================================================================

class GeminiVisionBackend(OCRBackend):
    """
    API-based document understanding using Google Gemini Vision.
    
    Best for:
    - Scanned pages
    - Complex layouts (multi-column, figures, tables)
    - When you need structured output
    
    Requires:
    - GOOGLE_API_KEY environment variable
    """
    
    name = "gemini_vision"
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        max_retries: int = 3,
    ):
        self.model = model
        self.max_retries = max_retries
        self._api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    def extract_pages(
        self,
        file_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Iterator[StructuredPage]:
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
        
        # Convert PDF pages to images
        images = self._pdf_to_images(file_path, page_range)
        
        for page_number, image_bytes in images:
            try:
                structured_page = self._process_page_image(page_number, image_bytes)
                yield structured_page
            except Exception as e:
                logger.warning(f"Gemini extraction failed for page {page_number}: {e}")
                # Yield empty page on failure
                yield StructuredPage(
                    page_number=page_number,
                    blocks=[],
                    ocr_backend=self.name,
                )
    
    def _pdf_to_images(
        self,
        file_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, bytes]]:
        """Convert PDF pages to images."""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("pdf2image not installed. Run: pip install pdf2image")
        
        # Determine page range
        first_page = page_range[0] if page_range else None
        last_page = page_range[1] if page_range else None
        
        # Convert at reasonable DPI for OCR
        images = convert_from_path(
            file_path,
            dpi=150,
            first_page=first_page,
            last_page=last_page,
        )
        
        result = []
        start_page = first_page or 1
        for idx, img in enumerate(images):
            # Convert to PNG bytes
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            result.append((start_page + idx, buffer.read()))
        
        return result
    
    def _process_page_image(
        self,
        page_number: int,
        image_bytes: bytes,
    ) -> StructuredPage:
        """Process a single page image with Gemini Vision."""
        import google.generativeai as genai
        
        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self.model)
        
        # Encode image
        image_b64 = base64.b64encode(image_bytes).decode()
        
        prompt = self._get_extraction_prompt()
        
        response = model.generate_content([
            {"mime_type": "image/png", "data": image_b64},
            prompt,
        ])
        
        # Parse response
        return self._parse_gemini_response(page_number, response.text)
    
    def _get_extraction_prompt(self) -> str:
        """Get the prompt for structured extraction."""
        return """Analyze this textbook page image and extract its structure.

Return a JSON object with:
{
  "blocks": [
    {
      "type": "heading_1|heading_2|heading_3|paragraph|equation|table|figure|caption|list_item",
      "text": "extracted text content",
      "latex": "LaTeX if equation, null otherwise",
      "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 50} or null
    }
  ],
  "is_scanned": true/false,
  "language": "en"
}

Guidelines:
- Preserve mathematical notation as LaTeX where possible
- Mark equations with type "equation"
- Identify heading levels (1=chapter, 2=section, 3=subsection)
- Include figure captions as "caption" type
- For tables, include the content as markdown in the "text" field
- Detect if page appears to be scanned vs digital

Return ONLY valid JSON, no other text."""
    
    def _parse_gemini_response(
        self,
        page_number: int,
        response_text: str,
    ) -> StructuredPage:
        """Parse Gemini response into StructuredPage."""
        import json
        
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text
            
            data = json.loads(json_str.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Gemini response as JSON for page {page_number}")
            return StructuredPage(
                page_number=page_number,
                blocks=[],
                raw_text=response_text,
                ocr_backend=self.name,
            )
        
        # Convert to PageBlocks
        blocks = []
        for idx, block_data in enumerate(data.get("blocks", [])):
            try:
                block_type = BlockType(block_data.get("type", "paragraph"))
            except ValueError:
                block_type = BlockType.PARAGRAPH
            
            bbox = None
            if block_data.get("bbox"):
                b = block_data["bbox"]
                bbox = BoundingBox(
                    x0=b.get("x0", 0),
                    y0=b.get("y0", 0),
                    x1=b.get("x1", 0),
                    y1=b.get("y1", 0),
                )
            
            blocks.append(PageBlock(
                block_type=block_type,
                text=block_data.get("text", ""),
                bbox=bbox,
                latex=block_data.get("latex"),
                order_index=idx,
            ))
        
        return StructuredPage(
            page_number=page_number,
            blocks=blocks,
            raw_text="\n".join(b.text for b in blocks),
            ocr_backend=self.name,
            is_scanned=data.get("is_scanned", False),
            language=data.get("language", "en"),
        )


# =============================================================================
# OpenAI Vision Backend (API-based)
# =============================================================================

class OpenAIVisionBackend(OCRBackend):
    """
    API-based document understanding using OpenAI GPT-4 Vision.
    
    Requires:
    - OPENAI_API_KEY environment variable
    """
    
    name = "openai_vision"
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3,
    ):
        self.model = model
        self.max_retries = max_retries
        self._api_key = os.getenv("OPENAI_API_KEY")
    
    def extract_pages(
        self,
        file_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Iterator[StructuredPage]:
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Reuse PDF to images from Gemini backend
        gemini_backend = GeminiVisionBackend()
        images = gemini_backend._pdf_to_images(file_path, page_range)
        
        for page_number, image_bytes in images:
            try:
                structured_page = self._process_page_image(page_number, image_bytes)
                yield structured_page
            except Exception as e:
                logger.warning(f"OpenAI extraction failed for page {page_number}: {e}")
                yield StructuredPage(
                    page_number=page_number,
                    blocks=[],
                    ocr_backend=self.name,
                )
    
    def _process_page_image(
        self,
        page_number: int,
        image_bytes: bytes,
    ) -> StructuredPage:
        """Process a single page image with OpenAI Vision."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self._api_key)
        
        # Encode image
        image_b64 = base64.b64encode(image_bytes).decode()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": self._get_extraction_prompt(),
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        
        response_text = response.choices[0].message.content
        return self._parse_response(page_number, response_text)
    
    def _get_extraction_prompt(self) -> str:
        """Same prompt as Gemini."""
        return GeminiVisionBackend()._get_extraction_prompt()
    
    def _parse_response(
        self,
        page_number: int,
        response_text: str,
    ) -> StructuredPage:
        """Parse OpenAI response - same format as Gemini."""
        # Reuse Gemini parser
        return GeminiVisionBackend()._parse_gemini_response(page_number, response_text)


# =============================================================================
# Hybrid Backend (Auto-selects based on page content)
# =============================================================================

class HybridOCRBackend(OCRBackend):
    """
    Hybrid backend that uses PyMuPDF for text-layer PDFs
    and falls back to vision API for scanned/complex pages.
    """
    
    name = "hybrid"
    
    def __init__(
        self,
        vision_backend: str = "gemini",
        scanned_threshold: float = 0.3,
    ):
        self.pymupdf = PyMuPDFBackend()
        
        if vision_backend == "gemini":
            self.vision = GeminiVisionBackend()
        elif vision_backend == "openai":
            self.vision = OpenAIVisionBackend()
        else:
            self.vision = None
        
        self.scanned_threshold = scanned_threshold
    
    def extract_pages(
        self,
        file_path: str,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Iterator[StructuredPage]:
        # First pass with PyMuPDF
        pymupdf_pages = list(self.pymupdf.extract_pages(file_path, page_range))
        
        for page in pymupdf_pages:
            # Check if page needs vision API
            if page.is_scanned and self.vision:
                try:
                    # Re-extract with vision
                    vision_pages = list(self.vision.extract_pages(
                        file_path,
                        page_range=(page.page_number, page.page_number),
                    ))
                    if vision_pages:
                        yield vision_pages[0]
                        continue
                except Exception as e:
                    logger.warning(f"Vision fallback failed for page {page.page_number}: {e}")
            
            yield page


# =============================================================================
# Backend Factory
# =============================================================================

def get_ocr_backend(
    backend_name: Optional[str] = None,
    **kwargs,
) -> OCRBackend:
    """
    Get an OCR backend by name.
    
    Args:
        backend_name: One of "pymupdf", "gemini", "openai", "hybrid"
                     If None, uses OCR_BACKEND env var or defaults to "hybrid"
        **kwargs: Additional arguments for the backend
    
    Returns:
        OCRBackend instance
    """
    if backend_name is None:
        backend_name = os.getenv("OCR_BACKEND", "hybrid")
    
    backend_name = backend_name.lower()
    
    if backend_name == "pymupdf":
        return PyMuPDFBackend(**kwargs)
    elif backend_name in ("gemini", "gemini_vision"):
        return GeminiVisionBackend(**kwargs)
    elif backend_name in ("openai", "openai_vision"):
        return OpenAIVisionBackend(**kwargs)
    elif backend_name == "hybrid":
        return HybridOCRBackend(**kwargs)
    else:
        raise ValueError(f"Unknown OCR backend: {backend_name}")
