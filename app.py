import io
import re
import base64
from typing import List, Dict, Set, Tuple

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

# ==========
# Helpers
# ==========

WELD_REGEX = re.compile(r"\b(?:SW|FW)\s*-?\s*(\d{1,4})\b", re.IGNORECASE)
PREFIX_REGEX = re.compile(r"\b(SW|FW)\s*-?\s*(\d{1,4})\b", re.IGNORECASE)

def normalize_weld_id(raw: str, zero_pad: int = 3) -> str:
    """
    Normalizes variants like 'sw 7', 'SW-07', 'fw12' -> 'SW-007' or 'FW-012'.
    """
    m = PREFIX_REGEX.search(raw)
    if not m:
        # try just digits with a best-guess prefix fallback
        m2 = re.search(r"\b(\d{1,4})\b", raw)
        if not m2:
            return ""
        return f"SW-{int(m2.group(1)):0{zero_pad}d}"
    prefix = m.group(1).upper()
    num = int(m.group(2))
    return f"{prefix}-{num:0{zero_pad}d}"

def image_bytes_from_page(page: fitz.Page, zoom: float = 3.0, rotate_deg: int = 0) -> bytes:
    mat = fitz.Matrix(zoom, zoom).preRotate(rotate_deg)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def extract_vector_text_welds(page: fitz.Page) -> Set[str]:
    """
    Gets text from vector layer (no OCR) and extracts weld IDs via regex.
    """
    results: Set[str] = set()
    # 'text' is fast; 'dict' gives spans, but 'text' usually suffices for labels
    text = page.get_text("text")
    for match in re.finditer(r"(SW|FW)\s*-?\s*(\d{1,4})", text, flags=re.IGNORECASE):
        raw = match.group(0)
        norm = normalize_weld_id(raw)
        if norm:
            results.add(norm)
    return results

def call_openai_vision(api_key: str, image_png_bytes: bytes) -> List[str]:
    """
    Calls GPT-4o-mini for robust OCR of weld tags from an image.
    Returns a list of strings that look like weld IDs (e.g., 'SW-012', 'FW-101').
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Encode image
    b64 = base64.b64encode(image_png_bytes).decode("utf-8")
    img_url = f"data:image/png;base64,{b64}"

    system = (
        "You read engineering isometric drawings. "
        "Find all weld identifiers on the page. "
        "A weld identifier is typically SW### or FW### (e.g., SW-012, FW 45). "
        "Return ONLY a JSON array of strings with the unique weld IDs you can see. "
        "No explanations."
    )
    user = (
        "Detect all weld IDs (like SW### or FW###) from this image. "
        "Please normalise to SW-### or FW-### with zero-padding to 3 digits if needed."
    )

    # Use responses API (vision)
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

    text_out = resp.choices[0].message.content or "[]"
    # Attempt to parse array of strings
    # Keep robust: pull SW/FW patterns from the response in case it isn't pure JSON
    found = set(re.findall(r"(?:SW|FW)\s*-?\s*(\d{1,4})", text_out, flags=re.IGNORECASE))
    welds = []
    for n in found:
        # we don't know the prefix from this simple regex; try to recover from full text
        # safer approach: search pairs to recover prefix+number
        for m in re.finditer(r"(SW|FW)\s*-?\s*(\d{1,4})", text_out, flags=re.IGNORECASE):
            prefix = m.group(1).upper()
            num = int(m.group(2))
            welds.append(f"{prefix}-{num:03d}")
    return sorted(set(welds))

def merge_sets(*sets: Set[str]) -> List[str]:
    all_items: Set[str] = set()
    for s in sets:
        all_items.update(s)
    return sorted(all_items)

def extract_welds_from_pdf(pdf_bytes: bytes,
                           aggressive: bool = False,
                           api_key: str = "",
                           zoom_levels: List[float] = [2.5, 3.0],
                           rotations: List[int] = [0, 90, 180, 270],
                           zero_pad: int = 3) -> List[str]:
    """
    Two-phase approach:
    1) Vector text pass (fast, very accurate when labels are vector).
    2) Optional aggressive pass: multi-scale + rotation + GPT-vision OCR, merge & dedupe.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    collected: Set[str] = set()

    # PHASE 1: Vector text
    for page in doc:
        vec_welds = extract_vector_text_welds(page)
        collected.update(vec_welds)

    if not aggressive:
        # Done
        return sorted(collected)

    # PHASE 2: Vision fallback
    if not api_key.strip():
        # If aggressive is on but no key, just return what we got from vector text.
        return sorted(collected)

    for page in doc:
        for rot in rotations:
            for z in zoom_levels:
                try:
                    png = image_bytes_from_page(page, zoom=z, rotate_deg=rot)
                    ocr_welds = call_openai_vision(api_key, png)
                    # Normalize again to be safe
                    for w in ocr_welds:
                        norm = normalize_weld_id(w, zero_pad=zero_pad)
                        if norm:
                            collected.add(norm)
                except Exception as e:
                    # Keep going; don't break extraction because one scale failed
                    print(f"[warn] vision pass failed at zoom {z}, rot {rot}: {e}")

    return sorted(collected)

def to_dataframe(weld_ids: List[str]) -> pd.DataFrame:
    # Split into columns: Prefix (SW/FW) + Number
    rows: List[Dict[str, str]] = []
    for w in weld_ids:
        m = re.match(r"^(SW|FW)-(\d{1,4})$", w)
        if m:
            rows.append({
                "Weld Number": w,
                "Joint Type": m.group(1),   # You can improve mapping if you store true joint type elsewhere
                "Joint Size (ND)": "",      # Filled elsewhere if you join with BOM
                "Material Description": ""  # Filled elsewhere if you join with spec/BOM
            })
        else:
            rows.append({
                "Weld Number": w,
                "Joint Type": "",
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
    aggressive = st.checkbox("Aggressive mode (better recall)", value=True,
                             help="Uses multi-scale, multi-rotation, and GPT‑Vision to catch tiny/rotated tags.")
    api_key = st.text_input("OpenAI API Key (for aggressive mode)", type="password")
    zoom_text = st.text_input("Zoom levels (comma‑separated)", value="2.5,3.0",
                              help="Higher zoom increases recall but costs more tokens/time in aggressive mode.")
    rotation_text = st.text_input("Rotations (degrees, comma‑separated)", value="0,90,180,270",
                                  help="Include 90/180/270 to catch rotated drawings.")
    zero_pad = st.number_input("Zero pad digits", min_value=1, max_value=4, value=3,
                               help="Will normalise SW-7 -> SW-007, etc.")

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
    with st.spinner("Extracting welds..."):
        pdf_bytes = uploaded_pdf.read()
        weld_ids = extract_welds_from_pdf(
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
    **Notes**
    - If a few welds are missing, try enabling **Aggressive mode**, increasing **Zoom levels** (e.g., `2.5,3.0,4.0`),
      and ensuring all **Rotations** are included.
    - Keep requirements lean for faster startup.
    """
)
