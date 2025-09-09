"""
Module for extracting text from PDF files.

We first attempt to use pdfminer.six to extract text per page. If that fails
for any reason, we fall back to PyPDF (pypdf) which has simpler extraction
capabilities but can still handle many documents. Returning a list of strings
per page helps identify which pages may lack a proper text layer, allowing
later stages to decide whether to send those pages for OCR.
"""

from io import BytesIO
from typing import List

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from pypdf import PdfReader



def _pdfminer_text_per_page(pdf_bytes: BytesIO) -> List[str]:
    """Return list of text strings, one per page, using pdfminer.six."""
    texts: List[str] = []
    for page_layout in extract_pages(pdf_bytes):
        chunks = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                chunks.append(element.get_text())
        texts.append("".join(chunks))
    return texts


def _pypdf_text_per_page(pdf_bytes: BytesIO) -> List[str]:
    """Fallback using pypdf if pdfminer fails."""
    reader = PdfReader(pdf_bytes)
    out: List[str] = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
    return out


def extract_text_per_page(pdf_file: BytesIO) -> List[str]:
    """
    Try to extract text from each page of a PDF.

    - We read the PDF into memory once, since both pdfminer and pypdf
      consume the stream.
    - pdfminer is used first as it can recover layout better. If that fails
      (raises an error), we fall back to pypdf.
    """
    data = pdf_file.read()
    try:
        return _pdfminer_text_per_page(BytesIO(data))
    except Exception:
        pass
    try:
        return _pypdf_text_per_page(BytesIO(data))
    except Exception:
        return []
