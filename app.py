import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import pandas as pd
import math
import io
import numpy as np

# 🚀 Load the OCR model once at the top (this is the fix!)
reader = easyocr.Reader(['en'], gpu=False)

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
