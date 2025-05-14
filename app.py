import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import pandas as pd
import math
import io
import numpy as np

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def pdf_page_to_image(pdf_bytes, page_number=0, zoom=2):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return img

def auto_assign_welds_to_bom(image, df_weld_types, df_bom, max_distance_threshold=150):
    reader = easyocr.Reader(['en'])
    img_np = np.array(image)
    results = reader.readtext(img_np)

    assignments = []
    weld_tags = []
    bom_tags = []

    for result in results:
        bbox, text, confidence = result
        text_clean = text.strip()
        if text_clean.isdigit():
            if 2 <= len(text_clean

