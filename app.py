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

# -----------------------------
# 1) OPENAI KEY LOADING (no more re-typing)
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

# OpenAI client that works with openai>=1.0
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    import openai
    openai.api_key = OPENAI_API_KEY
    client = None

# -----------------------------
# 2) UI – SIDEBAR
# -----------------------------
st.set_page_config(page_title="Weld Log Extraction (4 fields perfect)", layout="wide")
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
    """Render PDF pages into PIL images using pypdfium2 (no system deps)."""
    images: List[Image.Image] = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    n_pages = len(pdf)
    for i in range(n_pages):
        page = pdf[i]
        # scale controls DPI-ish. 2.0 is good, 2.5–3.0 for tiny text.
        pil = page.render(scale=scale).to_pil()
        images.append(pil.convert("RGB"))
    return images

def pil_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

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
- weld_size: the joint size (e.g., 25mm, DN25, 1"). If multiple sizes appear globally (e.g., BOM), use the specific size tied to each weld if shown; otherwise leave "" (do not guess).
- spec: the pipeline spec (e.g., CSJ, CSDN15). If one global spec clearly applies to all welds on the sheet, you may apply it; otherwise leave "" (no guessing).

RULES:
- Output EXACTLY what's visible. Never hallucinate rows or values.
- If you are reading an isometric, weld numbers are usually the tags near joints; return those only if legible.
- If information is missing or illegible, use empty string "" for that field but still include the weld row if the weld_number is present.
- Output JSON only, following this schema:

{"welds": [
  {"weld_number": "...", "shop_or_field": "Shop|Field|", "weld_size": "...", "spec": "..."},
  ...
]}

Do not include any other keys.
"""

def call_vision(images: List[Image.Image], model_name: str) -> Dict[str, Any]:
    """Send images to GPT-4o/4o-mini and get JSON back."""
    image_parts = [{"type": "input_image", "image_url": pil_to_base64_jpeg(img)} for img in images]

    # New OpenAI responses API
    if client is not None:
        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": image_parts}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        text = resp.output_text
    else:
        # Fallback for old SDKs
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": image_parts},
        ]
        completion = openai.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            response_format={"type":"json_object"},
        )
        text = completion.choices[0].message.content

    try:
        data = json.loads(text)
        if "welds" not in data or not isinstance(data["welds"], list):
            return {"welds": []}
        return data
    except Exception:
        return {"welds": []}

# -----------------------------
# 5) Normalisation + Table build
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
            # discard rows without a weld number
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
    # Stable sort: numeric if possible, otherwise lexicographic
    def try_num(x):
        try:
            return float(str(x).strip())
        except:
            return None
    if not df.empty:
        tmp = df["Weld Number"].apply(try_num)
        if tmp.notna().any():
            df = df.assign(_sort=tmp.fillna(1e12)).sort_values("_sort").drop(columns=["_sort"])
        else:
            df = df.sort_values("Weld Number", key=lambda s: s.astype(str))
        df = df.reset_index(drop=True)
    return df

# -----------------------------
# 6) Accuracy check vs Excel
# -----------------------------
def read_ground_truth_excel(xls_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    # Expect columns like: Weld No., Shop / Field, Pipe Size, Spec.
    # We'll be forgiving on names.
    df = pd.read_excel(xls, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    # map common variants → our 4 fields
    rename_map = {
        "Weld No.": "Weld Number",
        "Weld No": "Weld Number",
        "Weld_Number": "Weld Number",
        "Shop / Field": "Shop / Field",
        "Pipe Size": "Weld Size",
        "Spec.": "Spec",
        "Spec": "Spec",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    keep = [c for c in ["Weld Number", "Shop / Field", "Weld Size", "Spec"] if c in df.columns]
    df = df[keep].copy()
    # clean + normalise
    if "Weld Number" in df.columns:
        df["Weld Number"] = df["Weld Number"].apply(lambda x: str(x).strip().replace(".0",""))
    if "Shop / Field" in df.columns:
        df["Shop / Field"] = df["Shop / Field"].apply(normalise_shop_field)
    for c in ["Weld Size", "Spec"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: "" if pd.isna(x) else str(x).strip())
    return df.dropna(how="all")

def compare_tables(pred: pd.DataFrame, truth: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if pred.empty or truth.empty:
        return pd.DataFrame(), {}
    # left join on Weld Number
    merged = truth.merge(pred, on="Weld Number", how="left", suffixes=(" (truth)", " (pred)"))
    # evaluate per field
    stats = {}
    for col in ["Shop / Field", "Weld Size", "Spec"]:
        tcol = f"{col} (truth)"
        pcol = f"{col} (pred)"
        if tcol in merged.columns and pcol in merged.columns:
            match = (merged[tcol].fillna("") == merged[pcol].fillna(""))
            stats[col] = round(100.0 * match.mean(), 1)
            merged[f"{col} ✓"] = match.map({True: "✓", False: "×"})
    return merged, stats

# -----------------------------
# 7) UI – FILE UPLOADS
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
# 8) MAIN – run vision + show table
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

    with st.spinner("Extracting welds with GPT-4o Vision..."):
        result_json = call_vision(all_images, model)
    df_pred = to_table(result_json)

    st.subheader("Results")
    if df_pred.empty:
        st.error("No welds found. Try a higher render scale (e.g., 2.5–3.0) and re-run.")
    else:
        st.dataframe(df_pred, use_container_width=True)

        # Download Excel
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

    # -------------------------
    # Accuracy check vs Excel
    # -------------------------
    if truth_file and not df_pred.empty:
        st.subheader("Accuracy Check (vs. your Excel)")
        df_truth = read_ground_truth_excel(truth_file.read())
        if df_truth.empty:
            st.warning("Could not read the expected columns from your Excel. Make sure it has Weld No., Shop / Field, Pipe Size, Spec.")
        else:
            comp, stats = compare_tables(df_pred, df_truth)
            if stats:
                st.write(
                    f"**Match Rates:** "
                    f"Shop/Field: {stats.get('Shop / Field', 0)}%  |  "
                    f"Weld Size: {stats.get('Weld Size', 0)}%  |  "
                    f"Spec: {stats.get('Spec', 0)}%"
                )
            st.dataframe(comp, use_container_width=True)
