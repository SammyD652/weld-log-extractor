import io
import re
import json
from typing import List, Dict, Any

import pdfplumber
import pandas as pd

# OCR deps (optional)
try:
    import pypdfium2 as pdfium
    import pytesseract
    from PIL import Image, ImageOps, ImageFilter
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- PDF TEXT EXTRACTION (vector text + optional OCR fallback) ----------

def _page_text_pdfplumber(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            txt = "\n".join(line.strip() for line in txt.splitlines())
            pages.append({"page": i, "text": txt})
    return pages

def _ocr_page(pdf_bytes: bytes, page_index: int, dpi: int = 300) -> str:
    if not OCR_AVAILABLE:
        return ""
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    page = doc[page_index]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale).to_pil()
    img = bitmap.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    config = "-l eng --oem 3 --psm 6"
    text = pytesseract.image_to_string(img, config=config)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text

def extract_pdf_text_per_page(pdf_bytes: bytes, enable_ocr_fallback: bool = False) -> List[Dict[str, Any]]:
    pages = _page_text_pdfplumber(pdf_bytes)
    out = []
    for p in pages:
        text = p["text"]
        if enable_ocr_fallback or len(text.strip()) < 30:
            ocr_text = _ocr_page(pdf_bytes, p["page"] - 1)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        out.append({"page": p["page"], "text": text})
    return out

# ---------- WELD ID DETECTION ----------

# Accept variants:
#  - W-12 / W 12 / W:12 / W_12 / WELD-12
#  - SW 001 / SW-1  (socket weld numbering seen on your PDF)
#  - BW 015 / BW-15 (butt weld numbering if used)
# We capture type and number so we can normalize like SW-1 / W-12
WELD_PATTERN = re.compile(
    r"\b(?:(W(?:ELD)?)|(SW)|(BW))\s*[-_:]?\s*(?:No\.|#)?\s*(\d{1,5}[A-Z]?)\b",
    flags=re.IGNORECASE,
)

FIELD_WELD_BLOCKERS = re.compile(r"\b(FW|F/W|FIELD\s*WELD|FIELDWELD)\b", flags=re.IGNORECASE)
STRICT_W_FORM = re.compile(r"^(?:W|SW|BW)-\d{1,5}[A-Z]?$", flags=re.IGNORECASE)

def _normalize_weld_id(prefix: str, raw_num: str) -> str:
    num = raw_num.upper()
    m = re.match(r"(\d{1,5})([A-Z]?)$", num)
    core, suf = (m.group(1), m.group(2)) if m else (num, "")
    # Normalize 001 -> 1
    try:
        n = int(core)
    except Exception:
        n = core
    pref = prefix.upper()
    if pref.startswith("W"):
        pref = "W"
    return f"{pref}-{n}{suf}"

def _scan_text_for_matches(text: str, page_no: int, lines: List[str], idx: int) -> List[Dict[str, Any]]:
    """
    Runs regex on:
      - current line
      - bigram: line[i] + ' ' + line[i+1]
      - trigram: line[i] + ' ' + line[i+1] + ' ' + line[i+2]
    This catches SHX-split labels like 'SW' on one line and '001' on the next.
    """
    out = []
    windows = []
    l0 = lines[idx]
    windows.append(l0)
    if idx + 1 < len(lines):
        windows.append(l0 + " " + lines[idx + 1])
    if idx + 2 < len(lines):
        windows.append(l0 + " " + lines[idx + 1] + " " + lines[idx + 2])

    for w in windows:
        for m in WELD_PATTERN.finditer(w):
            pref = m.group(1) or m.group(2) or m.group(3) or "W"
            num = m.group(4)
            ctx_lines = lines[max(0, idx - 2): idx + 5]
            ctx = " | ".join(ctx_lines)
            out.append({
                "prefix": pref,
                "number": num,
                "page": page_no,
                "context": ctx
            })
    return out

def find_weld_candidates(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for p in pages:
        text = p["text"] or ""
        lines = text.split("\n")
        for idx, _ in enumerate(lines):
            found.extend(_scan_text_for_matches(text, p["page"], lines, idx))
    return found

def filter_and_normalize_welds(
    candidates: List[Dict[str, Any]],
    exclude_field_welds: bool,
    strict_form: bool
):
    rows = []
    for c in candidates:
        weld_id = _normalize_weld_id(c.get("prefix", "W"), c["number"])
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

    # De-duplicate deterministically
    seen = set()
    unique = []
    for r in rows:
        key = r["weld_id"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    # Sort by family (W < BW < SW), then number, then page
    def family_rank(wid: str) -> int:
        if wid.upper().startswith("W-"): return 0
        if wid.upper().startswith("BW-"): return 1
        if wid.upper().startswith("SW-"): return 2
        return 9

    def weld_num(wid: str) -> int:
        m = re.search(r"-([0-9]+)", wid)
        return int(m.group(1)) if m else 10**9

    unique.sort(key=lambda x: (family_rank(x["weld_id"]), weld_num(x["weld_id"]), x["page"]))
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

# ---------- EXPORT (safe even if empty) ----------

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        (df if df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="Weld Log")
        worksheet = writer.sheets["Weld Log"]
        for i, col in enumerate((df.columns if df is not None else [])):
            header_len = len(str(col))
            series = df[col].astype(str) if df is not None else pd.Series(dtype=str)
            mean_len = float(series.str.len().mean(skipna=True) or 0) if len(series) else 0
            width = min(50, max(12, int(round(max(header_len, mean_len) + 6))))
            worksheet.set_column(i, i, width)
    buf.seek(0)
    return buf.getvalue()
