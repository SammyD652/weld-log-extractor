"""
Module for performing OCR on scanned PDF pages.

For pages that lack a text layer, we convert them to images using
pdf2image.convert_from_bytes at a specified DPI, then feed those images
through Tesseract to obtain text. We return both the extracted text and
metrics such as average confidence per page for debugging purposes.
"""

from typing import Dict, List, Tuple
from io import BytesIO

from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image


def ocr_pages_needing_help(
    pdf_bytes: BytesIO,
    want_pages: List[int],
    dpi: int = 200,
    psm: int = 6,
) -> Tuple[Dict[int, str], Dict[int, dict]]:
    """
    OCR only specified pages from a PDF.

    Parameters
    ----------
    pdf_bytes : BytesIO
        In-memory PDF content.
    want_pages : List[int]
        Zero-based indices of pages to OCR.
    dpi : int, default 200
        Resolution to render the page images.
    psm : int, default 6
        Tesseract page segmentation mode.

    Returns
    -------
    Tuple[Dict[int, str], Dict[int, dict]]
        - Mapping of page index to OCR text.
        - Mapping of page index to diagnostic info: average confidence and number of words.
    """
    raw = pdf_bytes.read()
    # Render all pages once; we index into this list for efficiency
    images = convert_from_bytes(raw, dpi=dpi)
    out_texts: Dict[int, str] = {}
    report: Dict[int, dict] = {}

    for idx in want_pages:
        if idx < 0 or idx >= len(images):
            continue
        img: Image.Image = images[idx]
        config = f"--psm {psm}"
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT, config=config
        )
        words = data.get("text", [])
        confs = data.get("conf", [])
        # Filter out empty or punctuation-only tokens; join with spaces
        text = " ".join(w for w in words if w and w.strip() not in {"~", "|"})
        # Compute average confidence
        try:
            conf_vals = [float(c) for c in confs if c not in ("-1", "", None)]
            avg_conf = sum(conf_vals) / max(1, len(conf_vals))
        except Exception:
            avg_conf = 0.0
        out_texts[idx] = text
        report[idx] = {"ocr_conf_avg": round(avg_conf, 2), "lines": len(words)}
    return out_texts, report
