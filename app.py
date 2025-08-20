import io
import re
import base64
from typing import List, Dict, Set

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF

# ==========
# Regex + Normalisers
# ==========

PREFIX_REGEX = re.compile(r"\b(SW|FW)\s*-?\s*(\d{1,4})\b", re.IGNORECASE)

def normalize_weld_id(raw: str, zero_pad: int = 3) -> str:
    """
    Normalises variants like 'sw 7', 'SW-07', 'fw12' -> 'SW-007' or 'FW-012'.
    """
    m = PREFIX_REGEX.search(raw)
    if not m:
        # Fallback: digits only -> assume SW
        m2 = re.search(r"\b(\d{1,4})\b", raw)
        if not m2:
            return ""
        return f"SW-{int(m2.group(1)):0{zero_pad}d}"
    prefix = m.group(1).upper()
    num = int(m.group(2))
    return f"{prefix}-{num:0{zero_pad}d}"

# ==========
# PDF helpers
# ==========

def image_bytes_from_page(page: fitz.Page, zoom: float = 3.0, rotate_deg: int = 0) -> bytes:
    mat = fitz.Matrix(zoom, zoom).preRotate(rotate_deg)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def extract_vector_text_welds(page: fitz.Page) -> Set[str]:
    """
    Fast: pulls vector text from PDF and regex-matches weld IDs.
    """
    results: Set[str] = set()
    text = page.get_text("text")
    for m in re.finditer(r"(SW|FW)\s*-?\s*(\d{1,4})", text, flags=re.IGNORECASE):
        norm = normalize_weld_id(m.group(0))
        if norm:
            results.add(norm)
    return results

# ==========
# OpenAI Vision fallback (aggressive mode)
# ==========

def call_openai_vision(api_key: str, image_png_bytes: bytes) -> List[str]:
    """
    Calls GPT‑4o‑mini Vision to OCR weld tags from an image.
    Returns a de-duplicated list like ['SW-012','FW-045'].
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    b64 = base64.b64encode(image_png_bytes).decode("utf-8")
    img_url = f"data:image/png;base64,{b64}"

    system = (
        "You read engineering isometric drawings. "
        "Find all weld identifiers on the page. "
        "A weld identifier looks like SW### or FW###. "
        "Return ONLY a JSON array of strings, normalised as SW-### or FW-### with 3-digit padding."
    )
    user = "Detect and list all weld IDs in this image."

    # Using Chat Completions (vision via image_url)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            },
        ],
        temperature=0.1,
    )

    text_out = (resp.choices[0].message.content or "").strip()

    # Be robust: extract prefix+number pairs even if not pristine JSON
    found: Set[str] = set()
    for m in re.finditer(r"(SW|FW)\s*-?\s*(\d{1,4})", text_out, flags=re.IGNORECASE):
        found.add(f"{m.group(1).upper()}-{int(m.group(2)):03d}")
    return sorted(found)

def merge_sets(*sets: Set[str]) -> List[str]:
    all_items: Set[str] = set()
    for s in sets:
        all_items.update(s)
    return sorted(all_items)

@st.cache_data(show_spinner=False)
def extract_welds_from_pdf_cached(pdf_bytes: bytes,
                                  aggressive: bool,
                                  api_key: str,
                                  zoom_levels: List[float],
                                  rotations: List[int],
                                  zero_pad: int) -> List[str]:
    """
    Cached wrapper so re-running on the same file/settings is instant.
    """
    return extract_welds_from_pdf(pdf_bytes, aggressive, api_key, zoom_levels, rotations, zero_pad)

def extract_welds_from_pdf(pdf_bytes: bytes,
                           aggressive: bool = False,
                           api_key: str = "",
                           zoom_levels: List[float] = [2.5, 3.0],
                           rotations: List[int] = [0, 90, 180, 270],
                           zero_pad: int = 3) -> List[str]:
    """
    Two-phase approach:
      1) Vector text (fast, accurate when labels are text).
      2) Aggressive: multi-scale+rotation images -> GPT Vision OCR -> merge de-dupes.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    collected: Set[str] = set()

    # Phase 1: vector text
    for page in doc:
        collected.update(extract_vector_text_welds(page))

    # Done if not aggressive (or no key)
    if not aggressive or not api_key.strip():
        return sorted(collected)

    # Phase 2: Vision fallback
    for page in doc:
        for rot in rotations:
            for z in zoom_levels:
                try:
                    png = image_bytes_from_page(page, zoom=z, rotate_deg=rot)
                    for w in call_openai_vision(api_key, png):
                        norm = normalize_weld_id(w, zero_pad=zero_pad)
                        if norm:
                            collected.add(norm)
                except Exception as e:
                    # Keep going
                    print(f"[warn] vision pass failed z={z} rot={rot}: {e}")

    return sorted(collected)

def to_dataframe(weld_ids: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for w in weld_ids:
        m = re.match(r"^(SW|FW)-(\d{1,4})$", w)
        rows.append({
            "Weld Number": w,
            "Joint Type": m.group(1) if m else "",
            "Joint Size (ND)": "",
            "Material Description": ""
        })
    return pd.DataFrame(rows)

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Weld Log")
    return output.getvalue()

# ==========
# UI
# ==========

st.set_page_config(page_title="Weld Log Extractor", layout="centered")
st.title("📄➡️📊 Weld Log Extractor")

with st.sidebar:
    st.header("⚙️ Settings")
    aggressive = st.checkbox(
        "Aggressive mode (better recall)",
        value=True,
        help="Uses multi-scale, multi-rotation, and GPT‑Vision to catch tiny/rotated tags."
    )
    api_key = st.text_input("OpenAI API Key (for aggressive mode)", type="password")

    zoom_text = st.text_input(
        "Zoom levels (comma‑separated)",
        value="2.5,3.0",
        help="Add 4.0 if tiny labels are missed (slower/more tokens)."
    )
    rotation_text = st.text_input(
        "Rotations (degrees, comma‑separated)",
        value="0,90,180,270",
        help="Include all to catch rotated drawings."
    )
    zero_pad = st.number_input(
        "Zero pad digits",
        min_value=1, max_value=4, value=3,
        help="Normalises SW-7 -> SW-007."
    )

    def parse_floats(s: str) -> List[float]:
        out = []
        for x in s.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                out.append(float(x))
            except:
                pass
        return out or [3.0]

    def parse_ints(s: str) -> List[int]:
        out = []
        for x in s.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                out.append(int(x))
            except:
                pass
        return out or [0]

    zoom_levels = parse_floats(zoom_text)
    rotations = parse_ints(rotation_text)

uploaded_pdf = st.file_uploader("Upload Isometric PDF", type=["pdf"])

if st.button("Extract Weld Log", type="primary") and uploaded_pdf:
    with st.spinner("Extracting welds…"):
        pdf_bytes = uploaded_pdf.read()
        weld_ids = extract_welds_from_pdf_cached(
            pdf_bytes=pdf_bytes,
            aggressive=aggressive,
            api_key=api_key,
            zoom_levels=zoom_levels,
            rotations=rotations,
            zero_pad=zero_pad
        )
        df = to_dataframe(weld_ids)

    st.success(f"Found {len(df)} welds")
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
- If a weld is missing, enable **Aggressive mode**, add a higher zoom (e.g., `4.0`), and keep all **Rotations**.
- Keep requirements lean for fast startup.
"""
)
