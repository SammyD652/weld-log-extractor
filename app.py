import io
import os
import json
import base64
import time
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import pypdfium2 as pdfium
from openai import OpenAI


# =========================
# Basic page config
# =========================
st.set_page_config(page_title="Weld Log Extraction – 4 fields", layout="wide")
st.title("🔧 Weld Log Extraction (PDF → Weld Number, Shop/Field, Weld Size, Spec)")

# Always read API key from Streamlit Secrets (never hardcode)
def get_client() -> OpenAI:
    key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("Missing `OPENAI_API_KEY` in Streamlit Secrets. See steps at the bottom panel.")
        st.stop()
    return OpenAI(api_key=key)


# =========================
# PDF → PIL images (no external binaries)
# =========================
def pdf_bytes_to_images(pdf_bytes: bytes, scale: float = 2.0) -> List[Image.Image]:
    """Render each page to a PIL image using pypdfium2."""
    images: List[Image.Image] = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        images.append(pil.convert("RGB"))
    return images


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =========================
# Normalisation helpers
# =========================
def normalize_shop_field(val: str) -> str:
    """
    Map SW/S/W/S.W./S-W → Shop; FW/F/W/F.W./F-W → Field.
    Empty or unknown → "" (never guess).
    """
    if not isinstance(val, str):
        return ""
    s = val.strip().upper().replace(".", "").replace("-", "").replace("_", "")
    if s in {"S", "SW", "W", "SHOP", "S/W"}:   # treat 'W' seen on some logs as Shop (as requested)
        return "Shop"
    if s in {"F", "FW", "FIELD", "F/W"}:
        return "Field"
    return ""  # unknown → empty string per your rule


def clean_row(rec: Dict) -> Dict:
    """Coerce and trim fields; never invent values."""
    return {
        "Weld Number": str(rec.get("Weld Number", "") or "").strip(),
        "Shop/Field": normalize_shop_field(rec.get("Shop/Field", "")),
        "Weld Size": str(rec.get("Weld Size", "") or "").strip(),
        "Spec": str(rec.get("Spec", "") or "").strip(),
    }


def dedupe_by_weld_number(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        wn = r.get("Weld Number", "")
        key = wn  # keep first appearance
        if key not in seen:
            out.append(r)
            seen.add(key)
    return out


# =========================
# OpenAI Vision call (new SDK)
# =========================
SYSTEM_MSG = (
    "You are a meticulous extraction engine for welding documents. "
    "Extract ONLY what is explicitly visible. If a field is not visible, return an empty string. "
    "Output must be valid JSON with the schema:\n"
    "{ 'welds': [ { 'Weld Number': str, 'Shop/Field': str, 'Weld Size': str, 'Spec': str } ] }"
)

USER_RULES = (
    "From these isometric drawing pages, extract a table with EXACTLY these 4 fields:\n"
    "1) Weld Number\n2) Shop/Field (normalize SW/S/W → 'Shop', FW/F/W → 'Field')\n"
    "3) Weld Size\n4) Spec\n"
    "Rules:\n"
    "- Do NOT guess. If a value is not present, use empty string \"\".\n"
    "- Return a single JSON object with key 'welds'. No comments.\n"
    "- Avoid duplicates (same Weld Number).\n"
)

def build_messages(page_images: List[Image.Image]) -> List[Dict]:
    user_parts: List[Dict] = [{"type": "text", "text": USER_RULES}]
    for im in page_images:
        user_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{pil_to_base64_png(im)}"}
        })
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_parts},
    ]


def with_retries(fn, attempts=3, base_delay=1.2):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_delay * (2 ** i))
    raise last


def extract_rows(client: OpenAI, images: List[Image.Image], model_name: str) -> List[Dict]:
    messages = build_messages(images)

    def _invoke():
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        if not isinstance(data, dict) or "welds" not in data or not isinstance(data["welds"], list):
            raise ValueError("Unexpected JSON format (missing 'welds' list).")
        rows = [clean_row(x) for x in data["welds"]]
        rows = dedupe_by_weld_number(rows)
        return rows

    return with_retries(_invoke, attempts=3, base_delay=1.0)


# =========================
# Comparison vs Excel (✓/× per field)
# =========================
WANT_COLS = ["Weld Number", "Shop/Field", "Weld Size", "Spec"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        cu = c.strip().lower()
        if "weld" in cu and "number" in cu:
            col_map[c] = "Weld Number"
        elif "shop" in cu or "field" in cu:
            col_map[c] = "Shop/Field"
        elif "size" in cu or "nd" in cu:
            col_map[c] = "Weld Size"
        elif "spec" in cu:
            col_map[c] = "Spec"
    d2 = df.rename(columns=col_map).copy()
    for need in WANT_COLS:
        if need not in d2.columns:
            d2[need] = ""
    d2 = d2[WANT_COLS]
    d2["Weld Number"] = d2["Weld Number"].astype(str).str.strip()
    d2["Weld Size"] = d2["Weld Size"].astype(str).str.strip()
    d2["Spec"] = d2["Spec"].astype(str).str.strip()
    d2["Shop/Field"] = d2["Shop/Field"].apply(normalize_shop_field)
    return d2


def compare_with_excel(extracted_df: pd.DataFrame, excel_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    gt = normalize_columns(excel_df).dropna(how="all")
    gt = gt[gt["Weld Number"] != ""].drop_duplicates(subset=["Weld Number"])

    ex = normalize_columns(extracted_df).drop_duplicates(subset=["Weld Number"])

    merged = gt.merge(ex, on="Weld Number", how="left", suffixes=("_GT", "_EX"))
    rows = []
    for _, r in merged.iterrows():
        rows.append({
            "Weld Number": r["Weld Number"],
            "Shop/Field ✓?": "✓" if (r["Shop/Field_GT"] or "") == (r["Shop/Field_EX"] or "") else "×",
            "Weld Size ✓?": "✓" if (r["Weld Size_GT"] or "") == (r["Weld Size_EX"] or "") else "×",
            "Spec ✓?": "✓" if (r["Spec_GT"] or "") == (r["Spec_EX"] or "") else "×",
            "Shop/Field (GT)": r["Shop/Field_GT"] or "",
            "Shop/Field (EX)": r.get("Shop/Field_EX", "") or "",
            "Weld Size (GT)": r["Weld Size_GT"] or "",
            "Weld Size (EX)": r.get("Weld Size_EX", "") or "",
            "Spec (GT)": r["Spec_GT"] or "",
            "Spec (EX)": r.get("Spec_EX", "") or "",
        })
    cmp_df = pd.DataFrame(rows)

    total = max(1, 3 * len(cmp_df))
    correct = (cmp_df["Shop/Field ✓?"] == "✓").sum() + (cmp_df["Weld Size ✓?"] == "✓").sum() + (cmp_df["Spec ✓?"] == "✓").sum()
    return cmp_df, correct / total


# =========================
# UI: sidebar + uploads
# =========================
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Vision model",
        options=["gpt-4o-mini", "gpt-4o"],  # both support vision chat.completions + JSON
        index=0,
        help="Use gpt-4o for best accuracy if drawings are dense."
    )
    scale = st.slider("PDF render scale", 1.5, 3.0, 2.0, 0.1)
    st.caption("Increase if small tags are missed. Larger scale = clearer but slower.")

uploaded_pdf = st.file_uploader("Upload one isometric PDF", type=["pdf"])
uploaded_excel = st.file_uploader("(Optional) Upload ground-truth Excel (e.g., 2063 Excel Weld log.xlsx)", type=["xlsx", "xls"])

go = st.button("▶️ Extract 4 Fields")


# =========================
# Run
# =========================
if go:
    if not uploaded_pdf:
        st.warning("Please upload a PDF first.")
        st.stop()

    st.info("Rendering PDF pages to images…")
    try:
        page_images = pdf_bytes_to_images(uploaded_pdf.read(), scale=scale)
    except Exception as e:
        st.error(f"Failed to render PDF: {e}")
        st.stop()
    if not page_images:
        st.error("No pages found in the PDF.")
        st.stop()
    st.success(f"Rendered {len(page_images)} page(s).")

    client = get_client()

    st.info("Calling the vision model (no guessing, JSON-only)…")
    try:
        rows = extract_rows(client, page_images, model)
    except Exception as e:
        st.error(
            "Model call failed. This app uses the NEW `OpenAI` SDK with `chat.completions.create(...)` "
            "and JSON response_format. Remove any old `openai.ChatCompletion.create` usage.\n\n"
            f"Error: {e}"
        )
        st.stop()

    df = pd.DataFrame(rows, columns=["Weld Number", "Shop/Field", "Weld Size", "Spec"])
    st.subheader("Extracted Weld Log (4 fields)")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Extracted CSV",
        df.to_csv(index=False).encode("utf-8"),
        "extracted_weld_log.csv",
        "text/csv"
    )

    if uploaded_excel is not None:
        st.markdown("---")
        st.subheader("Comparison vs Ground Truth (✓/× per field)")
        try:
            gt = pd.read_excel(uploaded_excel)
        except Exception as e:
            st.error(f"Could not read the Excel file: {e}")
            st.stop()
        cmp_df, acc = compare_with_excel(df, gt)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        st.metric("Field-level Accuracy", f"{acc*100:.1f}%")
        st.caption("✓ means exact match with the Excel value for that Weld Number. Missing data is ''.")
    else:
        st.info("Tip: Upload your Excel to see ✓/× comparisons here.")

st.markdown("---")
with st.expander("How to set your API key in Streamlit Cloud"):
    st.markdown(
        """
**Where to click:**
1. In the top-right of your deployed app page click **⋮ (three dots)** → **Settings** → **Secrets**.
2. Paste:
```
OPENAI_API_KEY = "sk-..."
```
3. **Save**, then **⋮ → Reboot** the app.

**Notes**
- Uses new SDK: `from openai import OpenAI` → `client.chat.completions.create(...)`.
- `response_format={"type": "json_object"}` forces valid JSON.
- Temperature = 0 to avoid hallucinations.
- Normalises: `SW/S/W → Shop`, `FW/F/W → Field`.
- Missing data → `""` (never guessed).
        """
    )
