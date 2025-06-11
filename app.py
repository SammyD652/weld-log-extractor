import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import pandas as pd
import math
import io
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

# --- Helper Functions ---

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def pdf_page_to_image(pdf_bytes, page_number=0, zoom=2):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return img

def extract_text_data(image):
    bounds = reader.readtext(np.array(image))
    data = []
    for bbox, text, conf in bounds:
        x_center = int((bbox[0][0] + bbox[2][0]) / 2)
        y_center = int((bbox[0][1] + bbox[2][1]) / 2)
        data.append({
            "text": text.strip(),
            "x": x_center,
            "y": y_center
        })
    return pd.DataFrame(data)

def filter_weld_tags(df):
    return df[df["text"].str.match(r"^(SW|FW)?\s*\d{3}$", case=False, na=False)].copy()

def filter_bom_entries(df):
    return df[df["text"].str.contains(r"\bND\b", na=False)].copy()

def auto_assign_welds_to_bom(welds_df, bom_df, max_distance=100):
    assigned = []
    for _, weld in welds_df.iterrows():
        weld_pos = (weld['x'], weld['y'])
        closest = None
        min_dist = float('inf')

        for _, bom in bom_df.iterrows():
            bom_pos = (bom['x'], bom['y'])
            dist = distance(weld_pos, bom_pos)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                closest = bom

        if closest is not None:
            assigned.append({
                "Weld Tag": weld['text'],
                "Joint Info": closest['text'],
                "Distance": round(min_dist, 2)
            })
        else:
            assigned.append({
                "Weld Tag": weld['text'],
                "Joint Info": "Not found",
                "Distance": None
            })

    return pd.DataFrame(assigned)

# --- Streamlit App ---

st.set_page_config(page_title="Weld Log Extractor", layout="wide")
st.title("🔧 Weld Log Extractor (Beta)")

uploaded_file = st.file_uploader("Upload an isometric PDF drawing", type=["pdf"])

if uploaded_file:
    image = pdf_page_to_image(uploaded_file.read(), page_number=0, zoom=2)
    st.image(image, caption="Page 1 of uploaded drawing", use_column_width=True)

    with st.spinner("Running OCR..."):
        df = extract_text_data(image)
        weld_tags_df = filter_weld_tags(df)
        bom_df = filter_bom_entries(df)
        assigned_df = auto_assign_welds_to_bom(weld_tags_df, bom_df)

    st.subheader("📌 Detected Weld Tags")
    st.dataframe(weld_tags_df)

    st.subheader("📋 Detected BOM Entries (ND lines)")
    st.dataframe(bom_df)

    st.subheader("🔗 Auto-Matched Weld Log")
    st.dataframe(assigned_df)

    csv = assigned_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Weld Log CSV", csv, "weld_log.csv", "text/csv")

    st.success("Weld log extracted and matched successfully!")

else:
    st.info("Please upload a PDF drawing to get started.")
