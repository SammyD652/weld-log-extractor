import io
import os
import re
import base64
import json
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import pypdfium2 as pdfium
from openai import OpenAI

# -----------------------------
# 1) OPENAI KEY (no retyping)
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
                help="Tip: put it once in Streamlit → Settings → Secrets as OPENAI_API_KEY."
            )
    if not OPENAI_KEY:
        st.stop()
    return OPENAI_KEY

get_openai_api_key()
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
    render_scale = st.slider("PDF render scale (clarity vs speed)", 1.5, 3.0, 3.0, 0.1)
    st.caption("If small tags are missed, increase to 3.0 and re-run.")

# -----------------------------
# 3) PDF → PIL images
# -----------------------------
def pdf_bytes_to_images(pdf_bytes: bytes, scale: float = 3.0) -> List[Image.Image]:
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
# 4) Vision prompts (STRICT)
# -----------------------------
SYSTEM_PROMPT = '''
You are an expert welding QA/QC document reader.

You will be given one or more page images from either:
 - a mechanical isometric drawing, OR
 - a typed daily weld log.

TASK: Extract a weld list with ONLY these fields:
- weld_number: the weld identifier (exactly as shown). If not visible, DO NOT invent it.
- shop_or_field:
    * If the drawing/log explicitly marks "FW", "F/W", or "Field", return "Field".
    * Otherwise, default to "Shop".
- weld_size: the joint size (e.g., 25mm, DN25, 1"). If size cannot be definitively tied to a weld, use "" (no guessing).
- spec: the pipeline spec (e.g., CSJ, CSDN15). If a single global spec clearly applies to all welds on the sheet, you may apply it; otherwise use "".

CRITICAL RULES:
1) ONLY return weld numbers that are clearly visible as W-tags (e.g., W1, W01, W123) placed next to a drawn joint on the isometric.
2) IGNORE text from title blocks, borders, legends, revision notes, BOM/spec tables, general notes, and schedules. Do NOT read W-tags out of tables or legends.
3) DO NOT expand ranges (e.g., "W01–W50"). Only return tags you literally read as individual labels near joints.
4) Never classify as Field unless "FW/F/W/Field" is clearly printed near that weld tag. If uncertain, default to "Shop".
5) If any field is missing/illegible, use "" for that field, but still include the weld if weld_number is present.

Return JSON only in this schema:
{"welds":[
  {"weld_number":"...", "shop_or_field":"Shop|Field", "weld_size":"...", "spec":"..."}
]}

Additional constraints:
- A valid weld_number MUST look like W followed by 1–4 digits (e.g., W1, W01, W1234).
- If you are not sure the text is a weld tag next to a joint, do not include it.
'''

GLOBAL_SIZE_PROMPT = '''
You will be given one or more isometric page images. Decide if there is a SINGLE pipe size used across the sheet (e.g., ND 25, 25mm, DN25, 1").
- Check BOM/ND tables, title block, and repeated size notes.
- If there are multiple different sizes, return empty.
- If exactly one size dominates the entire sheet, return it as text (e.g., "25", "DN25", "25mm", or "1\"").

Return JSON ONLY:
{"single_size": "<text or empty string>"}
'''

# -----------------------------
# 5) OpenAI calls
# -----------------------------
def call_chat_completions(images: List[Image.Image], sys_prompt: str, user_intro_text: str) -> str:
    parts = [{"type": "text", "text": user_intro_text}]
    for img in images:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64(img)}"}
        })
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": parts},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""

def call_vision_extract(images: List[Image.Image]) -> Dict[str, Any]:
    text = call_chat_completions(images, SYSTEM_PROMPT, "Extract only the 4 fields and return JSON exactly as specified.")
    # Parse JSON or JSON inside text
    for attempt in range(2):
        try:
            data = json.loads(text) if attempt == 0 else json.loads(text[text.find("{"):text.rfind("}")+1])
            if isinstance(data, dict) and isinstance(data.get("welds", []), list):
                return data
        except Exception:
            pass
    return {"welds": []}

def detect_global_size(images: List[Image.Image]) -> str:
    """Ask the model if there's exactly one pipe size on this sheet."""
    text = call_chat_completions(images, GLOBAL_SIZE_PROMPT, "Determine if there is one single pipe size across the sheet. Return JSON.")
    try:
        data = json.loads(text)
        size = str(data.get("single_size", "")).strip()
        # quick cleanup: remove trailing punctuation
        size = size.rstrip(".,; ")
        return size
    except Exception:
        try:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start:end+1])
                return str(data.get("single_size", "")).strip().rstrip(".,; ")
        except Exception:
            pass
    return ""

# -----------------------------
# 6) Normalisation + validation + table
# -----------------------------
WELD_TAG_RE = re.compile(r"^[Ww]\d{1,4}$")  # W1..W9999

def normalise_shop_field(val: str) -> str:
    if not val:
        return ""
    v = str(val).strip().lower()
    if v in ("shop", "sw", "s/w"):
        return "Shop"
    if v in ("field", "fw", "f/w"):
        return "Field"
    return str(val).strip().title()

def is_valid_weld_number(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    return bool(WELD_TAG_RE.match(s))

def to_table(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for w in data.get("welds", []):
        wn = str(w.get("weld_number", "")).strip()
        if not is_valid_weld_number(wn):
            continue
        rows.append({
            "Weld Number": wn.upper(),
            "Shop / Field": normalise_shop_field(w.get("shop_or_field", "")),
            "Weld Size": str(w.get("weld_size", "")).strip(),
            "Spec": str(w.get("spec", "")).strip(),
        })

    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        return df

    # Safety cap: if absurdly high, assume hallucination; keep most plausible subset
    MAX_WELDS = 50
    if len(df) > MAX_WELDS:
        st.warning(f"Model returned {len(df)} welds (unusually high). Keeping the {MAX_WELDS} most plausible tags.")
        df["_score"] = (df["Weld Size"].astype(str).str.len() > 0).astype(int) + (df["Spec"].astype(str).str.len() > 0).astype(int)
        df = df.sort_values(["_score", "Weld Number"], ascending=[False, True]).head(MAX_WELDS).drop(columns=["_score"])

    # Sort by numeric part
    def tag_num(s: str) -> int:
        try:
            return int(re.sub(r"^[Ww]", "", s))
        except Exception:
            return 10**9
    df = df.sort_values(by="Weld Number", key=lambda s: s.map(tag_num)).reset_index(drop=True)
    return df

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
        df["Weld Number"] = df["Weld Number"].apply(lambda x: str(x).strip().upper().replace(".0", ""))
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
        result = call_vision_extract(all_imgs)
    df_pred = to_table(result)

    # -------- Global size fallback (fill blanks if the sheet uses a single size) --------
    if not df_pred.empty:
        blanks = df_pred["Weld Size"].astype(str).str.strip().eq("").sum()
        if blanks > 0:
            single_size = detect_global_size(all_imgs)
            if single_size:
                df_pred["Weld Size"] = df_pred["Weld Size"].apply(lambda x: single_size if str(x).strip() == "" else x)

    st.subheader("Results")
    if df_pred.empty:
        st.error("No welds found. Increase PDF render scale (3.0) and re-run.")
    else:
        st.dataframe(df_pred, use_container_width=True)

        # Download
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
