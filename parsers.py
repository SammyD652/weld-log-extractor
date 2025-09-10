import io
import re
import json
from typing import List, Dict, Any, Optional

import pdfplumber
import pandas as pd

# OCR deps
try:
    import pypdfium2 as pdfium
    import pytesseract
    from PIL import Image, ImageOps, ImageFilter
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- PDF TEXT EXTRACTION (deterministic with optional OCR fallback) ----------

def _page_text_pdfplumber(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            txt = "\n".join(line.strip() for line in txt.splitlines())
            pages.append({"page": i, "text": txt})
    return pages

def _ocr_page(pdf_bytes: bytes, page_index: int, dpi: int = 300) -> str:
    """
    Render a page to bitmap with pypdfium2, then OCR with Tesseract.
    Deterministic given same input & params.
    """
    if not OCR_AVAILABLE:
        return ""
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    page = doc[page_index]
    # scale factor: pixels = points * scale ; points are 72 dpi
    scale = dpi / 72.0
    bitmap = page.render(scale=scale).to_pil()  # PIL.Image
    # Light preprocessing for OCR
    img = bitmap.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    # Tesseract config: English, LSTM (oem 3), assume blocks of text (psm 6)
    config = "-l eng --oem 3 --psm 6"
    text = pytesseract.image_to_string(img, config=config)
    # Normalize whitespace
    text = "\n".join(line.strip() for line in text.splitlines())
    return text

def extract_pdf_text_per_page(pdf_bytes: bytes, enable_ocr_fallback: bool = False) -> List[Dict[str, Any]]:
