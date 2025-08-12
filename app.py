import os
import io
import base64
import json
from typing import List

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import streamlit as st
from openai import OpenAI

# --- Streamlit UI ---
st.set_page_config(page_title="Weld Log Extractor — GPT-4o Vision", layout="wide")
st.title("Weld Log Extractor — GPT-4o Vision")
st.caption("Upload your isometric drawing PDF (full drawing, no snips).")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Vision model", ["gpt-4o", "gpt-4o-mini"], index=0)
    max_pages = st.number_input("Max PDF pages to read", min_value=1, max_value=20, value=3, step=1)
    dpi = st.slider("Image render DPI", 100, 300, 220)
    xlsx_name = st.text_input("Excel file name", value="weld_log.xlsx")

# --- Helpers ---
def get_api_key() -> str:
    # First try Streamlit secrets, then env vars
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")

def pdf_pages_to_images(pdf_bytes: bytes, max_pages: int, dpi: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i in range(min(len(doc), max_pages)):
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return images

def pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def build_vision_messages(images: List[Image.Image]) -> list:
    content = [
        {
            "type": "text",
            "text": (
                "You are a welding QA assistant. Extract a structured weld log from these "
                "isometric drawing pages.\n\n"
                "For each weld, capture at least: Weld No/Tag, Line No, Size/Schedule, "
                "Material, Joint Type, NDT requirement, Notes. Output a compact JSON array "
                "of objects (one per weld). If not visible, leave fields empty.\n"
            ),
        }
    ]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{pil_to_base64(img)}"}
        })
    return [{"role": "user", "content": content}]

def json_to_dataframe(json_text: str) -> pd.DataFrame:
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "welds" in data:
            return pd.DataFrame(data["welds"])
    except Exception:
        pass
    return pd.DataFrame(columns=["Weld", "Line", "Size", "Schedule", "Material", "Joint", "NDT", "Notes"])

# --- Main app ---
pdf_file = st.file_uploader("PDF file", type=["pdf"])
if pdf_file:
    st.caption(f"Selected file: {pdf_file.name}")

    if st.button("Preview (first few pages)"):
        images = pdf_pages_to_images(pdf_file.read(), max_pages=int(max_pages), dpi=int(dpi))
        preview_cols = st.columns(2)
        for i, img in enumerate(images):
            preview_cols[i % 2].image(img, caption=f"Page {i+1}", use_column_width=True)

    if st.button("Extract Weld Log"):
        api_key = get_api_key()
        if not api_key:
            st.error("No API key found. Please add it to Streamlit Secrets.")
        else:
            images = pdf_pages_to_images(pdf_file.read(), max_pages=int(max_pages), dpi=int(dpi))
            messages = build_vision_messages(images)

            # Init client (no proxies)
            client = OpenAI(api_key=api_key)

            with st.spinner("Extracting weld log…"):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=2000,
                    )
                    raw_text = resp.choices[0].message.content.strip()
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")
                    st.stop()

            df = json_to_dataframe(raw_text)
            if df.empty:
                st.warning("Model returned no structured results. See raw output below.")
                st.code(raw_text)
            else:
                st.success(f"Found {len(df)} weld(s).")
                st.dataframe(df, use_container_width=True)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Weld Log")
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name=xlsx_name or "weld_log.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
else:
    st.info("Upload a PDF to begin.")

st.caption("Tip: Save your API key in Streamlit Secrets to avoid typing it each time.")
