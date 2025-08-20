import io
import re
import json
import base64
from typing import List, Dict, Set, Tuple, Optional

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image

# =========================
# Flexible patterns
# =========================
# Accept: SW-012, FW 12, sw007, 44, 44A  (prefix optional; suffix letter optional)
FLEX_PATTERN = re.compile(r"\b(?:(SW|FW)\s*[-]?\s*)?(\d{1,4})([A-Za-z])?\b", re.IGNORECASE)

def normalize_parts(prefix: Optional[str],
                    num_str: str,
                    suffix: Optional[str],
                    zero_pad: int = 3,
                    allow_no_prefix: bool = True,
                    default_prefix: str = "SW") -> str:
    """Normalise to PREFIX-###[SUFFIX]."""
    if not num_str.isdigit():
        m = re.match(r"(\d{1,4})([A-Za-z])?$", num_str)
        if m:
            num_str = m.group(1)
            suffix = suffix or m.group(2)
    p = (prefix or "").upper()
    if not p:
        if not allow_no_prefix:
            return ""
        p = default_prefix
    n = int(num_str)
    sfx = (suffix or "").upper()
    return f"{p}-{n:0{zero_pad}d}{sfx}"

# =========================
# PDF helpers
# =========================
def image_bytes_from_page(page: fitz.Page, zoom: float = 3.0, rotate_deg: int = 0) -> bytes:
    mat = fitz.Matrix(zoom, zoom).preRotate(rotate_deg)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def extract_vector_text_welds(page: fitz.Page,
                              zero_pad: int,
                              allow_no_prefix: bool,
                              default_prefix: str) -> Set[str]:
    results: Set[str] = set()
    text = page.get_text("text") or ""
    for m in re.finditer(FLEX_PATTERN, text):
        norm = normalize_parts(m.group(1), m.group(2), m.group(3),
                               zero_pad=zero_pad,
                               allow_no_prefix=allow_no_prefix,
                               default_prefix=default_prefix)
        if norm:
            results.add(norm)
    return results

# =========================
# OpenAI Vision fallback
# =========================
def call_openai_vision(api_key: str, image_png_bytes: bytes) -> str:
    """Return RAW model text (we’ll parse separately)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_png_bytes).decode("utf-8")
    img_url = f"data:image/png;base64,{b64}"
    system = (
        "You read engineering isometric drawings. "
        "Extract ALL weld identifiers you can see. "
        "Identifiers may be SW###, FW###, or just numbers like 44 or 44A. "
        "Respond with ONLY a JSON array. Each item: "
        "{\"raw\": string, \"prefix\": \"SW\"|\"FW\"|null, \"number\": string, \"suffix\": string|null}."
    )
    user = "Return only the JSON array. No commentary."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user","content":[
                {"type":"text","text":user},
                {"type":"image_url","image_url":{"url":img_url}},
            ]},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def parse_vision_output(raw_text: str,
                        zero_pad: int,
                        allow_no_prefix: bool,
                        default_prefix: str) -> List[str]:
    found: Set[str] = set()
    # 1) Try strict JSON
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict): continue
                prefix = item.get("prefix")
                number = str(item.get("number", "")).strip()
                suffix = item.get("suffix")
                norm = normalize_parts(prefix, number, suffix,
                                       zero_pad=zero_pad,
                                       allow_no_prefix=allow_no_prefix,
                                       default_prefix=default_prefix)
                if norm: found.add(norm)
    except Exception:
        pass
    # 2) Fallback regex sweep
    for m in re.finditer(FLEX_PATTERN, raw_text):
        norm = normalize_parts(m.group(1), m.group(2), m.group(3),
                               zero_pad=zero_pad,
                               allow_no_prefix=allow_no_prefix,
                               default_prefix=default_prefix)
        if norm: found.add(norm)
    return sorted(found)

# =========================
# Core extraction
# =========================
def extract_welds_from_pdf(pdf_bytes: bytes,
                           aggressive: bool,
                           api_key: str,
                           zoom_levels: List[float],
                           rotations: List[int],
                           zero_pad: int,
                           allow_no_prefix: bool,
                           default_prefix: str,
                           debug: bool
                           ) -> Tuple[List[str], Dict[str, int], bytes, List[Tuple[int, float, int, str]]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    collected: Set[str] = set()
    page_hits: Dict[str, int] = {}
    vision_debug: List[Tuple[int, float, int, str]] = []

    # Phase 1: vector
    for i, page in enumerate(doc):
        vec = extract_vector_text_welds(page, zero_pad, allow_no_prefix, default_prefix)
        collected.update(vec)
        page_hits[f"Page {i+1} (vector)"] = len(vec)

    # First-page preview
    preview_png = b""
    try:
        preview_png = image_bytes_from_page(doc[0], zoom=zoom_levels[0], rotate_deg=rotations[0])
    except Exception:
        pass

    # Phase 2: Vision
    if aggressive and api_key.strip():
        for i, page in enumerate(doc):
            for rot in rotations:
                for z in zoom_levels:
                    try:
                        png = image_bytes_from_page(page, zoom=z, rotate_deg=rot)
                        raw = call_openai_vision(api_key, png)
                        vision_debug.append((i + 1, z, rot, raw[:500]))
                        parsed = parse_vision_output(raw, zero_pad, allow_no_prefix, default_prefix)
                        collected.update(parsed)
                    except Exception as e:
                        if debug:
                            vision_debug.append((i + 1, z, rot, f"[error] {e}"))

    return sorted(collected), page_hits, preview_png, vision_debug

def to_dataframe(weld_ids: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for w in weld_ids:
        m = re.match(r"^(SW|FW)-(\d{1,4})([A-Z])?$", w)
        rows.append({
            "Weld Number": w,
            "Joint Type": (m.group(1) if m else ""),
            "Joint Size (ND)": "",
            "Material Description": ""
        })
    return pd.DataFrame(rows)

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Weld Log")
    return output.getvalue()

# =========================
# UI
# =========================
st.set_page_config(page_title="Weld Log Extractor", layout="centered")
st.title("📄➡️📊 Weld Log Extractor")

with st.sidebar:
    st.header("⚙️ Settings")
    aggressive = st.checkbox("Aggressive mode (better recall)", value=True)

    default_secret = st.secrets.get("general", {}).get("OPENAI_API_KEY", "") if "general" in st.secrets else ""
    api_key = st.text_input("OpenAI API Key (for aggressive mode)", type="password", value=default_secret)

    zoom_text = st.text_input("Zoom levels (comma-separated)", value="2.5,3.0,4.0,5.0")
    rotation_text = st.text_input("Rotations (degrees, comma-separated)", value="0,90,180,270")
    zero_pad = st.number_input("Zero pad digits", min_value=1, max_value=4, value=3)

    allow_no_prefix = st.checkbox("Allow number-only tags (assume SW)", value=True,
                                  help="If labels are just numbers like 44 or 44A, treat them as SW-044/044A.")
    default_prefix = st.selectbox("Default prefix for number-only tags", options=["SW", "FW"], index=0)

    debug = st.checkbox("Debug mode", value=True)

    def parse_floats(s: str) -> List[float]:
        out = []
        for x in s.split(","):
            x = x.strip()
            if x:
                try: out.append(float(x))
                except: pass
        return out or [3.0]

    def parse_ints(s: str) -> List[int]:
        out = []
        for x in s.split(","):
            x = x.strip()
            if x:
                try: out.append(int(x))
                except: pass
        return out or [0]

    zoom_levels = parse_floats(zoom_text)
    rotations = parse_ints(rotation_text)

uploaded_pdf = st.file_uploader("Upload Isometric PDF", type=["pdf"])

if st.button("Extract Weld Log", type="primary") and uploaded_pdf:
    with st.spinner("Extracting welds…"):
        pdf_bytes = uploaded_pdf.read()
        weld_ids, hit_counts, preview_png, vision_rows = extract_welds_from_pdf(
            pdf_bytes=pdf_bytes,
            aggressive=aggressive,
            api_key=api_key,
            zoom_levels=zoom_levels,
            rotations=rotations,
            zero_pad=zero_pad,
            allow_no_prefix=allow_no_prefix,
            default_prefix=default_prefix,
            debug=debug
        )
        df = to_dataframe(weld_ids)

    st.success(f"Found {len(df)} welds")
    if debug:
        st.subheader("Per-page vector-text hits")
        st.write(hit_counts)
        if preview_png:
            st.subheader("OCR preview (first page, first zoom/rotation)")
            st.image(preview_png, caption="If labels look tiny here, keep zoom 4.0/5.0.")
        if vision_rows:
            st.subheader("Raw Vision Output (first 500 chars per pass)")
            for (pg, z, rot, raw) in vision_rows:
                st.caption(f"Page {pg} | zoom {z} | rot {rot}")
                st.code(raw or "", language="json")
        if len(df) == 0:
            st.info("Still zero found. Keep Aggressive ON, try zoom 5.0, and keep all rotations. Share this screen if it persists.")

    st.dataframe(df, use_container_width=True)

    excel_bytes = df_to_excel_bytes(df)
    st.download_button(
        label="Download Excel",
        data=excel_bytes,
        file_name="weld_log.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown(
    """
**Tips**
- This drawing likely uses number-only balloons. Leave **“Allow number-only tags”** ON.
- For tiny labels add zoom `5.0` and keep rotations `0,90,180,270`.
"""
)
