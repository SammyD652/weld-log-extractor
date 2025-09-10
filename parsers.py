import io
import re
import json
from typing import List, Dict, Any, Optional

import pdfplumber
import pandas as pd

# OCR deps
try:
    import pypdfium2 as pdfium
    import pytesseract
    from PIL import Image, ImageOps, ImageFilter
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- PDF TEXT EXTRACTION (deterministic with optional OCR fallback) ----------

def _page_text_pdfplumber(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            txt = "\n".join(line.strip() for line in txt.splitlines())
            pages.append({"page": i, "text": txt})
    return pages

def _ocr_page(pdf_bytes: bytes, page_index: int, dpi: int = 300) -> str:
    """
    Render a page to bitmap with pypdfium2, then OCR with Tesseract.
    Deterministic given same input & params.
    """
    if not OCR_AVAILABLE:
        return ""
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    page = doc[page_index]
    # scale factor: pixels = points * scale ; points are 72 dpi
    scale = dpi / 72.0
    bitmap = page.render(scale=scale).to_pil()  # PIL.Image
    # Light preprocessing for OCR
    img = bitmap.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    # Tesseract config: English, LSTM (oem 3), assume blocks of text (psm 6)
    config = "-l eng --oem 3 --psm 6"
    text = pytesseract.image_to_string(img, config=config)
    # Normalize whitespace
    text = "\n".join(line.strip() for line in text.splitlines())
    return text

def extract_pdf_text_per_page(pdf_bytes: bytes, enable_ocr_fallback: bool = False) -> List[Dict[str, Any]]:
    """
    First try vector text (pdfplumber). If page has too little text or
    OCR is forced, do OCR on that page.
    """
    pages = _page_text_pdfplumber(pdf_bytes)
    # If forced, OCR all pages; else OCR only pages with almost no text
    out = []
    for p in pages:
        text = p["text"]
        if enable_ocr_fallback or len(text.strip()) < 30:
            ocr_text = _ocr_page(pdf_bytes, p["page"] - 1)
            # Prefer OCR if it yields more text
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        out.append({"page": p["page"], "text": text})
    return out

# ---------- WELD ID DETECTION ----------

# Accept common variants:
#  - W-12, W-123, W 12, W:12, W_12, WELD-12, WELD 12
#  - optional single trailing letter (e.g. W-12A) which some shops use
WELD_PATTERN = re.compile(
    r"\b(?:W(?:ELD)?)\s*[-_:]?\s*(?:No\.|#)?\s*(\d{1,5}[A-Z]?)\b",
    flags=re.IGNORECASE,
)

# Blockers for field welds
FIELD_WELD_BLOCKERS = re.compile(r"\b(FW|F/W|FIELD\s*WELD|FIELDWELD)\b", flags=re.IGNORECASE)

STRICT_W_FORM = re.compile(r"^W-\d{1,5}[A-Z]?$")

def find_weld_candidates(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find raw candidates with small context windows.
    Returns list of dicts: {weld_id_raw, number, page, context}
    """
    found: List[Dict[str, Any]] = []
    for p in pages:
        text = p["text"] or ""
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            for m in WELD_PATTERN.finditer(line):
                num = m.group(1)
                raw = m.group(0)
                ctx_lines = lines[max(0, idx - 2): idx + 3]
                ctx = " | ".join(ctx_lines)
                found.append({
                    "weld_id_raw": raw,
                    "number": num,
                    "page": p["page"],
                    "context": ctx
                })
    return found

def _normalize_weld_id(raw_num: str) -> str:
    num = raw_num.upper()
    # separate trailing letter if exists
    m = re.match(r"(\d{1,5})([A-Z]?)$", num)
    if not m:
        return f"W-{num}"
    core, suffix = m.group(1), m.group(2)
    return f"W-{int(core)}{suffix}"

def filter_and_normalize_welds(
    candidates: List[Dict[str, Any]],
    exclude_field_welds: bool,
    strict_form: bool
):
    """
    - Normalize to W-<n>[A]
    - Exclude FW/Field Weld if requested (by context window)
    - De-duplicate deterministically (first occurrence)
    - Sort deterministically by numeric then suffix then page
    """
    rows = []
    for c in candidates:
        weld_id = _normalize_weld_id(c["number"])
        ctx = c["context"] or ""
        if exclude_field_welds and FIELD_WELD_BLOCKERS.search(ctx):
            continue
        if strict_form and not STRICT_W_FORM.match(weld_id):
            continue
        rows.append({
            "weld_id": weld_id,
            "page": c["page"],
            "context": ctx,
        })

    seen = set()
    unique = []
    for r in rows:
        key = r["weld_id"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    def weld_num(wid: str) -> int:
        m = re.match(r"W-(\d+)", wid)
        return int(m.group(1)) if m else 10**9

    def weld_suffix(wid: str) -> str:
        m = re.match(r"W-\d+([A-Z]?)$", wid)
        return m.group(1) if m else ""

    unique.sort(key=lambda x: (weld_num(x["weld_id"]), weld_suffix(x["weld_id"]), x["page"]))
    return unique

# ---------- OPTIONAL LLM ENRICHMENT (temp=0) ----------

def _build_llm_messages(weld_id: str, ctx: str) -> list:
    system = (
        "You extract weld attributes only from the provided text context. "
        "If unknown, return empty strings. Do NOT guess or invent values. "
        "Return JSON strictly matching the schema."
    )
    user = (
        f"Context snippet from an isometric drawing's text:\n\n{ctx}\n\n"
        f"Target weld: {weld_id}\n\n"
        "If size (ND), joint type, or material description are explicitly present near this weld, extract them. "
        "Otherwise, leave them blank."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def _openai_client(api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def enrich_with_llm_fields(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    client = _openai_client(api_key)
    if client is None or df.empty:
        return df

    schema = {
        "name": "weld_fields",
        "schema": {
            "type": "object",
            "properties": {
                "weld_number": {"type": "string"},
                "joint_size_nd": {"type": "string"},
                "joint_type": {"type": "string"},
                "material_description": {"type": "string"},
            },
            "required": ["weld_number", "joint_size_nd", "joint_type", "material_description"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    out_rows = []
    for _, row in df.iterrows():
        messages = _build_llm_messages(str(row["Weld Number"]), str(row.get("Context", "")))
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=messages,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            out_rows.append({
                "Weld Number": row["Weld Number"],
                "Joint Size (ND)": data.get("joint_size_nd", ""),
                "Joint Type": data.get("joint_type", ""),
                "Material Description": data.get("material_description", ""),
                "Source Page": row["Source Page"],
                "Context": row.get("Context", ""),
            })
        except Exception:
            out_rows.append({
                "Weld Number": row["Weld Number"],
                "Joint Size (ND)": "",
                "Joint Type": "",
                "Material Description": "",
                "Source Page": row["Source Page"],
                "Context": row.get("Context", ""),
            })

    return pd.DataFrame(out_rows)

# ---------- EXPORT (safe for empty DataFrames) ----------

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Always returns a valid XLSX, even if df is empty.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        (df if df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="Weld Log")
        worksheet = writer.sheets["Weld Log"]

        # Auto column width (robust to empty frames)
        for i, col in enumerate((df.columns if df is not None else [])):
            # Length of header
            header_len = len(str(col))
            # Mean length of values (safe even if empty)
            series = df[col].astype(str) if df is not None else pd.Series(dtype=str)
            if len(series) == 0:
                mean_len = 0
            else:
                # Use average of value lengths; ignore NaN safely
                mean_len = float(series.str.len().mean(skipna=True) or 0)
            width = min(50, max(12, int(round(max(header_len, mean_len) + 6))))
            worksheet.set_column(i, i, width)
    buf.seek(0)
    return buf.getvalue()
