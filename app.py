import os
import io
import base64
import json
from typing import List

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import streamlit as st


# ---------- Streamlit page ----------
st.set_page_config(page_title="Weld Log Extractor — GPT-4o Vision", layout="wide")
st.title("Weld Log Extractor — GPT-4o Vision")
st.caption("Upload your isometric drawing PDF (full drawing, no snips).")


# ---------- Sidebar controls ----------
with st.sidebar:
    st.subheader("1) API Key")
    st.caption("Use an API key from Streamlit secrets or environment.")
    api_key_input = st.text_input("OpenAI API Key", type="password", help="If not provided, we'll try st.secrets or the OPENAI_API_KEY environment variable.")

    st.subheader("2) Model & Pages")
    model = st.selectbox("Vision model", ["gpt-4o", "gpt-4o-mini"], index=0)
    max_pages = st.number_input("Max PDF pages to read", min_value=1, max_value=20, value=3, step=1)
    dpi = st.slider("Image render DPI (higher = sharper, slower)", min_value=100, max_value=300, value=220)

    st.subheader("3) Output")
    xlsx_name = st.text_input("Excel file name", value="weld_log.xlsx")


# ---------- Helpers ----------
def get_api_key() -> str:
    """Return API key from sidebar, st.secrets or env."""
    if api_key_input.strip():
        return api_key_input.strip()
    # Streamlit secrets (Streamlit Cloud)
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # Environment variable
    return os.getenv("OPENAI_API_KEY", "")


def clear_proxy_env():
    """Remove proxy env vars that break httpx/OpenAI in some hosts."""
    for k in [
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
        "ALL_PROXY", "all_proxy",
        "NO_PROXY", "no_proxy",
    ]:
        if k in os.environ:
            os.environ.pop(k, None)


def pdf_pages_to_images(pdf_bytes: bytes, max_pages: int = 3, dpi: int = 220) -> List[Image.Image]:
    """
    Render first N pages of a PDF as PIL images using PyMuPDF.
    """
    images: List[Image.Image] = []
    zoom = dpi / 72.0  # 72 dpi base
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = min(len(doc), max_pages)
        for page_index in range(total):
            pix = doc[page_index].get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


def pil_to_base64(img: Image.Image) -> str:
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def build_vision_messages(images: List[Image.Image]) -> list:
    """
    Build Chat Completions style messages containing base64 images + a task prompt.
    """
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
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{pil_to_base64(img)}"
                },
            }
        )
    messages = [{"role": "user", "content": content}]
    return messages


def json_to_dataframe(json_text: str) -> pd.DataFrame:
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and "welds" in data and isinstance(data["welds"], list):
            return pd.DataFrame(data["welds"])
    except Exception:
        pass
    # Fallback: empty frame
    return pd.DataFrame(columns=["Weld", "Line", "Size", "Schedule", "Material", "Joint", "NDT", "Notes"])


# ---------- Main UI ----------
pdf_file = st.file_uploader("PDF file", type=["pdf"], label_visibility="visible")
preview_btn = st.button("Preview (first few pages)")
run_btn = st.button("Extract Weld Log")

selected_file_label = ""
if pdf_file is not None:
    selected_file_label = f"Selected file: {pdf_file.name}"
else:
    selected_file_label = "No PDF selected"

st.caption(selected_file_label)

# ---------- Preview ----------
if pdf_file and preview_btn:
    st.info("Rendering PDF pages and generating previews… This can take ~10–30 seconds depending on DPI & pages.")
    pdf_bytes = pdf_file.read()
    try:
        preview_cols = st.columns(2)
        images = pdf_pages_to_images(pdf_bytes, max_pages=int(max_pages), dpi=int(dpi))
        # Display previews in two columns (Streamlit 1.38 uses use_column_width)
        for i, img in enumerate(images):
            preview_cols[i % 2].image(img, caption=f"Page {i+1}", use_column_width=True)
    except Exception as e:
        st.error(f"Preview failed: {e}")

# ---------- Extraction ----------
if pdf_file and run_btn:
    api_key = get_api_key()
    if not api_key:
        st.error("No OpenAI API key found. Enter it in the sidebar or set it in Streamlit secrets.")
        st.stop()

    # Convert PDF ➜ images
    pdf_bytes = pdf_file.read()
    images = pdf_pages_to_images(pdf_bytes, max_pages=int(max_pages), dpi=int(dpi))

    # Build messages with images
    messages = build_vision_messages(images)

    # Prepare client (remove proxies first)
    try:
        clear_proxy_env()
        os.environ["OPENAI_API_KEY"] = api_key
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialise OpenAI client: {e}")
        st.stop()

    # Call the model
    with st.spinner("Calling GPT‑4o Vision to extract weld log…"):
        try:
            # Using Chat Completions (works with openai>=1.0)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")
            st.stop()

    # Parse into DataFrame
    df = json_to_dataframe(text)
    if df.empty:
        st.warning("Model returned no structured results. Showing raw text below.")
        st.code(text)
        st.stop()

    # Show & download
    st.success(f"Extracted {len(df)} weld rows.")
    st.dataframe(df, use_container_width=True)

    # Write to Excel in-memory and offer download
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Weld Log")
    st.download_button(
        label="Download Excel",
        data=buff.getvalue(),
        file_name=xlsx_name or "weld_log.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("Tip: To avoid re-entering the API key on Streamlit Cloud, put it in `.streamlit/secrets.toml` as `OPENAI_API_KEY = \"sk-...\"`.")
