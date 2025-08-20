import io
import re
import base64
from typing import List, Dict, Set, Tuple

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image

# =========================
# Config / Regex
# =========================

WELD_PATTERN = r"(SW|FW)\s*[-]?\s*(\d{1,4})"
WELD_REGEX = re.compile(WELD_PATTERN, re.IGNORECASE)

def normalize_weld_id(raw: str, zero_pad: int = 3) -> str:
    """
    Normalises things like 'sw 7', 'SW-07', 'fw12' -> 'SW-007' or 'FW-012'.
    """
    m = WELD_REGEX.search(raw)
    if not m:
        # Fallback: if pure digits, assume SW
        m2 = re.search(r"\b(\d{1,4})\b", raw)
        if not m2:
            return ""
        return f"SW-{int(m2.group(1)):0{zero_pad}d}"
    prefix = m.group(1).upper()
    num = int(m.group(2))
    return f"{prefix}-{num:0{zero_pad}d}"

# =========================
# PDF helpers
# =========================

def image_bytes_from_page(page: fitz.Page, zoom: float = 3.0, rotate_deg: int = 0) -> bytes:
    mat = fitz.Matrix(zoom, zoom).preRotate(rotate_deg)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def extract_vector_text_welds(page: fitz.Page) -> Set[str]:
    """
    Vector text pass (fast). If text is embedded, this is very accurate.
    """
    results: Set[str] = set()
    text = page.get_text("text") or ""
    for m in re.finditer(WELD_PATTERN, text, flags=re.IGNORECASE):
        norm = normalize_weld_id(m.group(0))
        if norm:
            results.add(norm)
    return results

# =========================
# OpenAI Vision fallback
# =========================

def call_openai_vision(api_key: str, image_png_bytes: bytes) -> List[str]:
    """
    GPT‑4o‑mini Vision: robust OCR for tricky/rotated/tiny labels.
    Returns deduped list like ['SW-012','FW-045'].
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
    found: Set[str] = set()
    for m in re.finditer(WELD_PATTERN, text_out, flags=re.IGNORECASE):
        found.add(f"{m.group(1).upper()}-{int(m.group(2)):03d}")
    return sorted(found)

# =========================
# Core extraction
# =========================

def extract_welds_from_pdf(pdf_bytes: bytes,
                           aggressive: bool = False,
                           api_key: str = "",
                           zoom_levels: List[float] = [2.5, 3.0],
                           rotations: List[int] = [0, 90, 180, 270],
                           zero_pad: int = 3,
                           debug: bool = False
                           ) -> Tuple[List[str], Dict[str, int], bytes]:
    """
    Returns (weld_ids, page_hit_counts, preview_png_bytes)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    collected: Set[str] = set()
    page_hits: Dict[str, int] = {}

    # Phase 1: vector text
    for i, page in enumerate(doc):
        vec_welds = extract_vector_text_welds(page)
        collected.update(vec_welds)
        page_hits[f"Page {i+1} (vector)"] = len(vec_welds)

    # Prepare a small preview (first page, first zoom/rotation)
    preview_png = b""
    try:
        first_page = doc[0]
        preview_png = image_bytes_from_page(first_page, zoom=zoom_levels[0], rotate_deg=rotations[0])
    except Exception:
        pass

    # If not aggressive (or no key), stop here
    if not aggressive or not api_key.strip():
        return sorted(collected), page_hits, preview_png

    # Phase 2: Vision fallback
    for i, page in enumerate(doc):
        for rot in rotations:
            for z in zoom_levels:
                try:
                    png = image_bytes_from_page(page, zoom=z, rotate_deg=rot)
                    ocr_welds = call_openai_vision(api_key, png)
                    for w in ocr_welds:
                        norm = normalize_weld_id(w, zero_pad=zero_pad)
                        if norm:
                            collected.add(norm)
                except Exception as e:
                    if debug:
                        print(f"[warn] vision pass failed on page {i+1}, z={z}, rot={rot}: {e}")

    return sorted(collected), page_hits, preview_png

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

# =========================
# UI
# =========================

st.set_page_config(page_title="Weld Log Extractor", layout="centered")
st.title("📄➡️📊 Weld Log Extractor")

with st.sidebar:
    st.header("⚙️ Settings")
    aggressive = st.checkbox(
        "Aggressive mode (better recall)",
        value=True,
        help="Uses multi-scale, multi-rotation, and GPT‑Vision to catch tiny/rotated tags."
    )

    # Optional: read API key from secrets if you set .streamlit/secrets.toml
    default_secret = st.secrets.get("general", {}).get("OPENAI_API_KEY", "") if "general" in st.secrets else ""
    api_key = st.text_input("OpenAI API Key (for aggressive mode)", type="password", value=default_secret)

    zoom_text = st.text_input(
        "Zoom levels (comma‑separated)",
        value="2.5,3.0,4.0",
        help="Add 4.0 or 5.0 if labels are very small."
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
    debug = st.checkbox("Debug mode", value=True, help="Shows per-page hits and a preview image.")

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
        weld_ids, hit_counts, preview_png = extract_welds_from_pdf(
            pdf_bytes=pdf_bytes,
            aggressive=aggressive,
            api_key=api_key,
            zoom_levels=zoom_levels,
            rotations=rotations,
            zero_pad=zero_pad,
            debug=debug
        )
        df = to_dataframe(weld_ids)

    st.success(f"Found {len(df)} welds")
    if debug:
        # Per-page vector hits
        st.subheader("Per‑page vector‑text hits")
        st.write(hit_counts)
        if preview_png:
            st.subheader("OCR preview (first page with first zoom/rotation)")
            st.image(preview_png, caption="If labels look tiny here, keep zoom 4.0 or try 5.0.")

        # Hint if nothing found
        if len(df) == 0:
            st.info("No vector hits found and/or OCR found none. Enable Aggressive mode, add zoom 4.0 or 5.0, and keep all rotations. If still zero, send me this screen and the PDF so I can add another pattern.")

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
- If a weld is missing, enable **Aggressive mode**, add a higher zoom (e.g., `4.0` or `5.0`), and keep all **Rotations**.
- If you always get zero from the vector pass, your PDF likely has image‑only text; keep Aggressive mode on.
"""
)
