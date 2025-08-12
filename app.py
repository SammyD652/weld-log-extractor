import os
import io
import re
import json
import base64
from typing import List, Tuple

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import streamlit as st

# =========================
# UI
# =========================
st.set_page_config(page_title="Weld Log Extractor — PDF → Excel", layout="wide")
st.title("Weld Log Extractor — PDF → Excel")
st.caption("Upload the full isometric PDF. The app renders a few pages as images and asks GPT‑4o Vision to extract the weld log.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Vision model", ["gpt-4o", "gpt-4o-mini"], index=0)
    max_pages = st.number_input("Max PDF pages to read", min_value=1, max_value=15, value=3, step=1)
    dpi = st.slider("Image render DPI", 120, 300, 220)
    file_name = st.text_input("Excel file name", value="weld_log.xlsx")

# =========================
# Helpers
# =========================
def get_api_key() -> str:
    # Streamlit Cloud: set in Settings → Secrets: OPENAI_API_KEY = "sk-..."
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
        pages = min(len(doc), max_pages)
        for i in range(pages):
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def build_prompt_and_content(images: List[Image.Image]) -> list:
    """
    Strictly request the four fields required for your weld log.
    """
    instruction = (
        "You are a welding QA assistant. Extract a weld log from these isometric drawing pages.\n"
        "Return STRICT JSON ONLY (no commentary), an array of objects.\n"
        "Fields (exact keys):\n"
        '  "Weld Number", "Joint Size (ND)", "Joint Type", "Material Description"\n'
        "Rules:\n"
        "  • If a field isn’t visible, use an empty string.\n"
        "  • Do NOT include additional fields.\n"
        "  • Keep numeric strings as they appear (e.g., 'DN50', '2\"').\n"
        "  • Do not deduplicate; return all seen welds.\n"
    )
    content = [{"type": "text", "text": instruction}]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{pil_to_base64(img)}"}
        })
    return [{"role": "user", "content": content}]

def extract_json_block(text: str) -> str:
    """
    If the model returns extra prose, try to pull the first JSON array.
    """
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return text
    m = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.S)
    return m.group(0) if m else text

def to_dataframe(json_text: str) -> pd.DataFrame:
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            # Validate columns and coerce
            cols = ["Weld Number", "Joint Size (ND)", "Joint Type", "Material Description"]
            df = pd.DataFrame(data)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols]
    except Exception:
        pass
    # fallback empty
    return pd.DataFrame(columns=["Weld Number", "Joint Size (ND)", "Joint Type", "Material Description"])

# --- OpenAI call (works whether openai lib is present or not) ---
def call_openai(messages: list, model: str, api_key: str) -> Tuple[bool, str]:
    """
    Returns (ok, text). Uses 'openai' client if available, else raw HTTP via requests.
    """
    try:
        from openai import OpenAI  # Streamlit Cloud installs per requirements.txt
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
        )
        return True, resp.choices[0].message.content.strip()
    except Exception:
        try:
            import requests
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000,
            }
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            r.raise_for_status()
            data = r.json()
            return True, data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return False, f"OpenAI call failed: {e}"

# =========================
# Main
# =========================
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    st.caption(f"Selected: {pdf_file.name}")
    col1, col2 = st.columns(2)
    if col1.button("Preview pages"):
        imgs = pdf_pages_to_images(pdf_file.read(), max_pages=int(max_pages), dpi=int(dpi))
        show_cols = st.columns(2)
        for i, im in enumerate(imgs):
            show_cols[i % 2].image(im, caption=f"Page {i+1}", use_container_width=True)

    if col2.button("Extract Weld Log"):
        api_key = get_api_key()
        if not api_key:
            st.error("No API key found. In Streamlit Cloud: Settings → Secrets → add OPENAI_API_KEY.")
            st.stop()

        # Re-read file for extraction (uploader stream gets consumed by preview)
        pdf_bytes = pdf_file.getvalue()
        images = pdf_pages_to_images(pdf_bytes, max_pages=int(max_pages), dpi=int(dpi))
        messages = build_prompt_and_content(images)

        with st.spinner("Reading PDF, calling GPT‑4o Vision, and parsing…"):
            ok, raw = call_openai(messages, model, api_key)

        if not ok:
            st.error(raw)
            st.stop()

        raw_json = extract_json_block(raw)
        df = to_dataframe(raw_json)

        if df.empty:
            st.warning("No structured welds returned. Showing raw output for debugging below.")
            st.code(raw, language="json")
        else:
            st.success(f"Extracted {len(df)} weld(s).")
            st.dataframe(df, use_container_width=True)

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Weld Log")

            st.download_button(
                label="Download Excel",
                data=buf.getvalue(),
                file_name=file_name or "weld_log.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
else:
    st.info("Upload a PDF to begin.")
st.caption("Tip: In Streamlit Cloud, set OPENAI_API_KEY in Settings → Secrets so you don’t type it every time.")
