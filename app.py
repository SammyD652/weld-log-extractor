import io
import os
import base64
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import pypdfium2 as pdfium
import openai  # old SDK style

# -----------------------------
# 1) OPENAI KEY LOADING
# -----------------------------
def get_openai_api_key():
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        key = os.getenv("OPENAI_API_KEY", None)
    if not key:
        with st.sidebar:
            st.markdown("### OpenAI API Key")
            key = st.text_input(
                "Enter your OpenAI API key",
                type="password",
                help="Tip: Save this once in Streamlit → Settings → Secrets as OPENAI_API_KEY to avoid typing it again."
            )
    if not key:
        st.stop()
    return key

OPENAI_API_KEY = get_openai_api_key()
openai.api_key = OPENAI_API_KEY

# -----------------------------
# 2) UI – SIDEBAR
# -----------------------------
st.set_page_config(page_title="Weld Log Extraction – Focused 4 Fields", layout="wide")
st.title("Weld Log Extraction – Focused 4 Fields")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0,
                         help="Use 4o for best quality. 4o-mini is cheaper.")
    render_scale = st.slider("PDF render scale (clarity vs. speed)", 1.5, 3.0, 2.0, 0.1)
    st.caption("If small tags are missed, increase the scale and re-run.")

# -----------------------------
# 3) Helpers – PDF → PIL images
# -----------------------------
def pdf_bytes_to_images(pdf_bytes: bytes, scale: float = 2.0) -> List[Image.Image]:
    images: List[Image.Image] = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    n_pages = len(pdf)
    for i in range(n_pages):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        images.append(pil.convert("RGB"))
    return images

def pil_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64

# -----------------------------
# 4) Vision Prompt – ONLY 4 FIELDS
# -----------------------------
SYSTEM_PROMPT = """You are an expert welding QA/QC document reader.

You will be given one or more page images from either:
 - a mechanical isometric drawing, OR
 - a typed daily weld log.

TASK: Extract a weld list with ONLY these fields:
- weld_number: the weld identifier (as shown on drawing or log). If not visible, DO NOT invent it.
- shop_or_field: "Shop" or "Field". Map any of {SW, S/W, Shop} → "Shop". Map {FW, F/W, Field} → "Field". If unknown, use "".
- weld_size: the joint size (e.g., 25mm, DN25, 1"). If multiple sizes appear globally (e.g., BOM), use the specific size tied to each weld if shown; otherwise leave "".
- spec: the pipeline spec (e.g., CSJ, CSDN15). If one global spec clearly applies to all welds on the sheet, you may apply it; otherwise leave "".

RULES:
- Output EXACTLY what's visible. Never hallucinate rows or values.
- Output JSON only, following this schema:

{"welds": [
  {"weld_number": "...", "shop_or_field": "Shop|Field|", "weld_size": "...", "spec": "..."}
]}
"""

# -----------------------------
# 5) GPT Vision call (old SDK)
# -----------------------------
def call_vision(images: List[Image.Image], model_name: str) -> Dict[str, Any]:
    image_parts = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + pil_to_base64_jpeg(img)}}
        for img in images
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": image_parts},
    ]

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    text = completion["choices"][0]["message"]["content"]

    try:
        data = json.loads(text)
        if "welds" not in data or not isinstance(data["welds"], list):
            return {"welds": []}
        return data
    except Exception:
        return {"welds": []}

# -----------------------------
# 6) Normalisation + Table build
# -----------------------------
def normalise_shop_field(val: str) -> str:
    if not val:
        return ""
    v = str(val).strip().lower()
    if v in ["shop", "sw", "s/w"]:
        return "Shop"
    if v in ["field", "fw", "f/w"]:
        return "Field"
    return val.strip().title()

def to_table(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for w in data.get("welds", []):
        weld_number = str(w.get("weld_number", "")).strip()
        if not weld_number:
            continue
        shop_field = normalise_shop_field(w.get("shop_or_field", ""))
        weld_size  = str(w.get("weld_size", "")).strip()
        spec       = str(w.get("spec", "")).strip()
        rows.append({
            "Weld Number": weld_number,
            "Shop / Field": shop_field,
            "Weld Size": weld_size,
            "Spec": spec
        })
    df = pd.DataFrame(rows).drop_duplicates()
    return df.reset_index(drop=True)

# -----------------------------
# 7) File Upload UI
# -----------------------------
st.subheader("1) Upload PDF(s)")
pdf_files = st.file_uploader(
    "Drop one or more PDFs (isometrics or weld logs)",
    type=["pdf"],
    accept_multiple_files=True
)

st.subheader("2) (Optional) Upload Ground Truth Excel")
truth_file = st.file_uploader(
    "Drop your Excel containing the 4 fields",
    type=["xlsx", "xls"]
)

run = st.button("Run Extraction", type="primary", use_container_width=True)

# -----------------------------
# 8) Main run
# -----------------------------
if run:
    if not pdf_files:
        st.warning("Please upload at least one PDF first.")
        st.stop()

    all_images: List[Image.Image] = []
    with st.spinner("Rendering PDF pages..."):
        for f in pdf_files:
            images = pdf_bytes_to_images(f.read(), scale=render_scale)
            all_images.extend(images)

    st.success(f"Rendered {len(all_images)} page image(s).")

    with st.spinner("Extracting welds with GPT Vision..."):
        result_json = call_vision(all_images, model)
    df_pred = to_table(result_json)

    st.subheader("Results")
    if df_pred.empty:
        st.error("No welds found. Try a higher render scale (e.g., 2.5–3.0) and re-run.")
    else:
        st.dataframe(df_pred, use_container_width=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_pred.to_excel(writer, index=False, sheet_name="Weld Log (4 fields)")
        st.download_button(
            "Download Excel",
            data=buf.getvalue(),
            file_name="weld_log_4fields.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
