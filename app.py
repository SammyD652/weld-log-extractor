import io
import os
import base64
import json
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import pypdfium2 as pdfium

# -----------------------------
# 1) OPENAI (>=1.x) CLIENT
# -----------------------------
OPENAI_KEY = None
def get_openai_api_key():
    global OPENAI_KEY
    if OPENAI_KEY:
        return OPENAI_KEY
    if hasattr(st, "secrets"):
        OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None)
    if not OPENAI_KEY:
        OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)
    if not OPENAI_KEY:
        with st.sidebar:
            st.markdown("### OpenAI API Key")
            OPENAI_KEY = st.text_input(
                "Enter your OpenAI API key",
                type="password",
                help="Tip: put it in Streamlit → Settings → Secrets as OPENAI_API_KEY to avoid typing again."
            )
    if not OPENAI_KEY:
        st.stop()
    return OPENAI_KEY

get_openai_api_key()

from openai import OpenAI
client = OpenAI(api_key=OPENAI_KEY)

# -----------------------------
# 2) PAGE / SIDEBAR
# -----------------------------
st.set_page_config(page_title="Weld Log Extraction – Focused 4 Fields", layout="wide")
st.title("Weld Log Extraction – Focused 4 Fields")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="Use gpt-4o for best accuracy."
    )
    render_scale = st.slider("PDF render scale (clarity vs speed)", 1.5, 3.0, 2.5, 0.1)
    st.caption("If small tags are missed, increase scale to 2.5–3.0 and re-run.")

# -----------------------------
# 3) PDF → images
# -----------------------------
def pdf_bytes_to_images(pdf_bytes: bytes, scale: float = 2.5) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        imgs.append(pil.convert("RGB"))
    return imgs

def pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------------
# 4) Vision prompt
# -----------------------------
SYSTEM_PROMPT = """
You are an expert welding QA/QC document reader.

You will be given one or more page images from either:
 - a mechanical isometric drawing, OR
 - a typed daily weld log.

TASK: Extract a weld list with ONLY these fields:
- weld_number: the weld identifier (exactly as shown). If not visible, DO NOT invent it.
- shop_or_field: "Shop" or "Field". Map any of {SW, S/W, Shop} → "Shop". Map {FW, F/W, Field} → "Field". If unknown, use "".
- weld_size: the joint size (e.g., 25mm, DN25, 1"). If size cannot be definitively tied to a weld, use "" (no guessing).
- spec: the pipeline spec (e.g., CSJ, CSDN15). If a single global spec clearly applies to all welds on the sheet, you may apply it; otherwise use "".

RULES:
- Output EXACTLY what's visible. No hallucinations.
- If any field is missing/illegible, use "" for that field, but include the weld if the weld_number is present.

Return JSON only in this schema:
{"welds":[
  {"weld_number":"...", "shop_or_field":"Shop|Field|", "weld_size":"...", "spec":"..."}
]}
"""

# -----------------------------
# 5) Call GPT-4o (Chat Completions with image_url parts)
# -----------------------------
def call_vision(images: List[Image.Image], model_name: str) -> Dict[str, Any]:
    # Chat Completions requires content parts of type "text" and "image_url"
    parts = [{"type": "text", "text": "Extract only the 4 fields as per instructions and return JSON."}]
    for img in images:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64(img)}"}
        })

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": parts},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        # Fallback: Responses API (works with "input_image")
        try:
            img_parts = [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{pil_to_b64(img)}"}
                for img in images
            ]
            resp2 = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "input_text", "text": "Extract JSON only."}] + img_parts},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            text = resp2.output_text
        except Exception:
            text = ""

    # Try parse JSON directly
    try:
        data = json.loads(text)
        if "welds" in data and isinstance(data["welds"], list):
            return data
    except Exception:
        pass

    # Last resort: find JSON block inside text
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
            if "welds" in data and isinstance(data["welds"], list):
                return data
    except Exception:
        pass

    return {"welds": []}

# -----------------------------
# 6) Normalisation + table
# -----------------------------
def normalise_shop_field(val: str) -> str:
    if not val:
        return ""
    v = str(val).strip().lower()
    if v in ("shop", "sw", "s/w"):
        return "Shop"
    if v in ("field", "fw", "f/w"):
        return "Field"
    return str(val).strip().title()

def to_table(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for w in data.get("welds", []):
        wn = str(w.get("weld_number", "")).strip()
        if not wn:
            continue
        rows.append({
            "Weld Number": wn,
            "Shop / Field": normalise_shop_field(w.get("shop_or_field", "")),
            "Weld Size": str(w.get("weld_size", "")).strip(),
            "Spec": str(w.get("spec", "")).strip(),
        })
    df = pd.DataFrame(rows).drop_duplicates()
    return df.reset_index(drop=True)

# -----------------------------
# 7) Optional: read Excel truth & compare
# -----------------------------
def read_truth_excel(xls_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    df = pd.read_excel(xls, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "Weld No.": "Weld Number",
        "Weld No": "Weld Number",
        "Weld_Number": "Weld Number",
        "Shop / Field": "Shop / Field",
        "Pipe Size": "Weld Size",
        "Spec.": "Spec",
        "Spec": "Spec",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["Weld Number", "Shop / Field", "Weld Size", "Spec"] if c in df.columns]
    df = df[keep].copy()
    if "Weld Number" in df.columns:
        df["Weld Number"] = df["Weld Number"].apply(lambda x: str(x).strip().replace(".0", ""))
    if "Shop / Field" in df.columns:
        df["Shop / Field"] = df["Shop / Field"].apply(normalise_shop_field)
    for c in ("Weld Size", "Spec"):
        if c in df.columns:
            df[c] = df[c].apply(lambda x: "" if pd.isna(x) else str(x).strip())
    return df.dropna(how="all")

def compare(pred: pd.DataFrame, truth: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if pred.empty or truth.empty:
        return pd.DataFrame(), {}
    merged = truth.merge(pred, on="Weld Number", how="left", suffixes=(" (truth)", " (pred)"))
    stats = {}
    for col in ["Shop / Field", "Weld Size", "Spec"]:
        t, p = f"{col} (truth)", f"{col} (pred)"
        if t in merged.columns and p in merged.columns:
            hit = (merged[t].fillna("") == merged[p].fillna(""))
            merged[f"{col} ✓"] = hit.map({True: "✓", False: "×"})
            stats[col] = round(100.0 * hit.mean(), 1)
    return merged, stats

# -----------------------------
# 8) UI – uploads
# -----------------------------
st.subheader("1) Upload PDF(s)")
pdfs = st.file_uploader("Drop one or more PDFs (isometrics or weld logs)", type=["pdf"], accept_multiple_files=True)

st.subheader("2) (Optional) Upload Ground Truth Excel")
truth_file = st.file_uploader("Drop your Excel containing the 4 fields", type=["xlsx", "xls"])

run = st.button("Run Extraction", type="primary", use_container_width=True)

# -----------------------------
# 9) Main
# -----------------------------
if run:
    if not pdfs:
        st.warning("Please upload at least one PDF.")
        st.stop()

    all_imgs: List[Image.Image] = []
    with st.spinner("Rendering PDF pages…"):
        for f in pdfs:
            all_imgs.extend(pdf_bytes_to_images(f.read(), scale=render_scale))
    st.success(f"Rendered {len(all_imgs)} page image(s).")

    with st.spinner("Extracting welds with GPT-4o…"):
        result = call_vision(all_imgs, model)
    df_pred = to_table(result)

    st.subheader("Results")
    if df_pred.empty:
        st.error("No welds found. Increase PDF render scale (2.5–3.0) and re-run.")
    else:
        st.dataframe(df_pred, use_container_width=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            df_pred.to_excel(w, index=False, sheet_name="Weld Log (4 fields)")
        st.download_button(
            "Download Excel",
            data=buf.getvalue(),
            file_name="weld_log_4fields.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    if truth_file and not df_pred.empty:
        st.subheader("Accuracy Check (vs. your Excel)")
        df_truth = read_truth_excel(truth_file.read())
        if df_truth.empty:
            st.warning("Could not detect expected columns in your Excel.")
        else:
            comp, stats = compare(df_pred, df_truth)
            if stats:
                st.write(f"**Match Rates** – Shop/Field: {stats.get('Shop / Field',0)}% | "
                         f"Weld Size: {stats.get('Weld Size',0)}% | Spec: {stats.get('Spec',0)}%")
            st.dataframe(comp, use_container_width=True)
