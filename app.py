import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import numpy as np
import fitz  # PyMuPDF
import re

st.set_page_config(layout="centered")
st.title("Weld Log Drawing Reader")

uploaded_file = st.file_uploader("Upload Drawing (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

def pdf_to_image(pdf_bytes, page_number=0):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom x2 for clarity
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def preprocess_image(image):
    gray = image.convert("L")  # Grayscale
    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    cleaned = enhanced.filter(ImageFilter.MedianFilter(size=3))
    return cleaned

if uploaded_file:
    file_type = uploaded_file.type

    # Load image from file
    if file_type == "application/pdf":
        image = pdf_to_image(uploaded_file.read())
    else:
        image = Image.open(uploaded_file)

    st.image(image, caption="Drawing Preview", use_container_width=True)

    # Preprocess and OCR
    with st.spinner("Running OCR..."):
        prepped = preprocess_image(image)
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(np.array(prepped))

    # Filter weld tags
    weld_tags = []
    for (bbox, text, confidence) in result:
        cleaned_text = text.upper().replace(" ", "")
        if re.match(r'(SW|FW)\d{3}', cleaned_text):
            weld_tags.append((cleaned_text, confidence))

    if weld_tags:
        st.subheader("Detected Weld Tags:")
        for tag, conf in weld_tags:
            st.write(f"🔧 **{tag}** (Confidence: {conf:.2f})")
    else:
        st.warning("No weld tags detected. Try zooming in on the drawing or uploading a higher-resolution image.")
