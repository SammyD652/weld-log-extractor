import streamlit as st
from PIL import Image
import easyocr
import numpy as np

st.set_page_config(layout="centered")

st.title("Weld Log Drawing Reader")

# Upload an image
uploaded_file = st.file_uploader("Upload Drawing Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Drawing Preview", use_container_width=True)

    # Run OCR
    with st.spinner("Running OCR..."):
        reader = easyocr.Reader(['en'], gpu=False)  # Force CPU to avoid torch error
        result = reader.readtext(np.array(image))

    # Display OCR results
    st.subheader("Detected Text:")
    for (bbox, text, confidence) in result:
