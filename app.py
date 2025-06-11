import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import pandas as pd
import math
import io
import numpy as np

# Safe model loader with spinner
@st.cache_resource(show_spinner="Loading OCR model (this can take 1–2 mins)...")
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def pdf_page_to_image(pdf_bytes, page_number=0, zoom=2):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return img

def extract_weld_numbers(img):
    bounds = reader.readtext(np.array(img))
    results = []
    for bound in bounds:
        text = bound[1]
        if any(tag in text for tag in ["SW", "FW"]):
            results.append(text)
    return results

#
