import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import pandas as pd
import math
import io
import numpy as np
import re

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
    # Convert to grayscale and boost contrast
    img = img.convert("L")
    img = img.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = img.point(lambda x: 0 if x < 128 else 255, '1')

    # Run OCR
    text = pytesseract.image_to_string(img)

    # Find SW or FW followed by 3 digits
    matches = re.findall(r'(SW|FW)?\s*(\d{3})', text)

    results = []
    for weld_type, weld_no in matches:
        label = f"{weld_type.strip() if weld_type else ''}{weld_no}"
        results.append(label)

    return results

# Streamlit App
st.title("Weld Log Extractor")

uploaded_file = st.file_uploader("Upload a PDF drawing", type="pdf")

if uploaded_file is not None:
    page_num = st.number_input("Page number", min_value=1, step=1, value=1)
    st.write("Processing page:", page_num)

    image = pdf_page_to_image(uploaded_file.read(), page_number=page_num - 1)
    st.image(image, caption="Extracted Page")

    st.write("Extracting weld numbers...")
    welds = extract_weld_numbers(image)

    st.write("Weld numbers found:")
    st.write(welds)

    if welds:
        df = pd.DataFrame(welds, columns=["Weld Number"])
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "weld_log.csv", "text/csv")
