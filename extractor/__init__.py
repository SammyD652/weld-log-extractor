"""
Extractor package for weld log extraction.

This package includes modules for extracting text from PDFs, performing OCR on
scanned pages, parsing isometric drawings for weld data, optional LLM-based
validation, and pydantic schemas for structured records.
"""

__all__ = [
    "pdf_text",
    "pdf_ocr",
    "parse_iso",
    "llm_validate",
    "schema",
]
