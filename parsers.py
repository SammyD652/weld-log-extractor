import io, re, json
from typing import List, Dict, Any, Tuple

import pdfplumber
import pandas as pd

# Optional OCR dependencies
try:
    import pypdfium2 as pdfium
    import pytesseract
    from PIL import Image, ImageOps, ImageFilter
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------------- PDF Extraction ----------------

def _page_text_pdfplumber(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            txt = "\n".join(line.strip() for line in txt.splitlines())
            pages.append({"page": i, "text": txt})
    return pages

def _ocr_page(pdf_bytes: bytes, page_index: int, dpi: int = 300) -> str:
    """Render a page to image and OCR it with a character whitelist."""
    if not OCR_AVAILABLE:
        return ""
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    page = doc[page_index]
    img = page.render(scale=dpi / 72.0).to_pil().convert("L")
    img = ImageOps.autocontrast(img).filter(ImageFilter.SHARPEN)
    # PSM 6 = assume a single uniform block; good for diagrams
    config = "-l eng --oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_:#"
    text = pytesseract.image_to_string(img, config=config)
    return "\n".join(line.strip() for line in text.splitlines())

def extract_pdf_text_per_page(pdf_bytes: bytes, enable_ocr_fallback: bool = False) -> List[Dict[str, Any]]:
    """
    Always try vector text first; if a page is nearly empty (or user forces it),
    OCR that page and use whichever text is longer.
    """
    pages = _page_text_pdfplumber(pdf_bytes)
    out: List[Dict[str, Any]] = []
    for p in pages:
        text = p["text"] or ""
        if enable_ocr_fallback or len(text.strip()) < 30:
            ocr_text = _ocr_page(pdf_bytes, p["page"] - 1)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        out.append({"page": p["page"], "text": text})
    return out

# ---------------- Weld ID Detection ----------------

# Compact matcher (also works on compacted windows)
BASE = re.compile(
    r"\b(?:(SW|BW|W))(?:ELD)?[-_:\s]*(?:No\.|#)?\s*([0-9OIl]{1,5}[A-Z]?)\b",
    re.IGNORECASE,
)

FIELD_WELD_BLOCKERS = re.compile(r"\b(FW|F/W|FIELD\s*WELD|FIELDWELD)\b", re.IGNORECASE)
STRICT_W_FORM = re.compile(r"^(?:W|SW|BW)-\d{1,5}[A-Z]?$", re.IGNORECASE)

def _fix_digits(s: str) -> str:
    # correct common OCR digit confusions inside numbers
    return s.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")

def _normalize(prefix: str, raw: str) -> str:
    raw = _fix_digits(raw).upper()
    m = re.match(r"(\d{1,5})([A-Z]?)$", raw)
    if m:
        return f"{prefix.upper()}-{int(m.group(1))}{m.group(2)}"
    return f"{prefix.upper()}-{raw}"

def _stitch_ladders(lines: List[str]) -> List[str]:
    """
    Join sequences of single-char lines into one token: S, W, 0, 0, 1 -> SW001
    Keep originals too so context isn't lost.
    """
    stitched: List[str] = []
    buf: List[str] = []
    def flush():
        if buf:
            token = "".join(buf)
            stitched.append(token)
            stitched.append(re.sub(r"[^A-Za-z0-9:_#-]+", "", token))  # compact version
            buf.clear()
    for ln in lines:
        t = (ln or "").strip()
        if len(t) == 1 and re.match(r"[A-Za-z0-9]", t):
            buf.append(t)
        else:
            flush()
            stitched.append(ln)
    flush()
    return stitched

def _scan_windows(lines: List[str], idx: int, max_lines: int) -> List[str]:
    """
    Build multiple windows joining up to N lines and produce spaced + compact variants.
    """
    out: List[str] = []
    for k in range(1, max_lines + 1):
        if idx + k <= len(lines):
            seg = " ".join(lines[idx: idx + k])
            out.append(re.sub(r"\s+", " ", seg))
            out.append(re.sub(r"[^A-Za-z0-9:_#-]+", "", seg))
    return out

def find_weld_candidates(
    pages: List[Dict[str, Any]],
    aggressive: bool = True,
    return_preview: bool = False
) -> Tuple[List[Dict[str, Any]], List[str]]:
    found: List[Dict[str, Any]] = []
    preview: List[str] = []
    for p in pages:
        raw_lines = (p["text"] or "").split("\n")
        lines = _stitch_ladders(raw_lines) if aggressive else raw_lines
        max_join = 12 if aggressive else 1
        for idx in range(len(lines)):
            for w in _scan_windows(lines, idx, max_join):
                for m in BASE.finditer(w):
                    pref = (m.group(1) or "W").upper()
                    num = m.group(2)
                    ctx = " | ".join(lines[max(0, idx - 4): idx + 8])
                    found.append({"prefix": pref, "number": num, "page": p["page"], "context": ctx})
                    if return_preview:
                        # snapshot the matched string (helps you see exactly what matched)
                        preview.append(w)
    return (found, preview) if return_preview else (found, [])

def filter_and_normalize_welds(
    cands: List[Dict[str, Any]],
    exclude_field_welds: bool,
    strict_form: bool
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in cands:
        wid = _normalize(c["prefix"], c["number"])
        ctx = c.get("context", "") or ""
        if exclude_field_welds and FIELD_WELD_BLOCKERS.search(ctx):
            continue
        if strict_form and not STRICT_W_FORM.match(wid):
            continue
        rows.append({"weld_id": wid, "page": c["page"], "context": ctx})

    # De-dupe deterministically, then sort by family/number/page
    seen, uniq = set(), []
    for r in rows:
        if r["weld_id"] in seen:
            continue
        seen.add(r["weld_id"])
        uniq.append(r)

    def family_rank(wid: str) -> int:
        u = wid.upper()
        if u.startswith("W-"): return 0
        if u.startswith("BW-"): return 1
        if u.startswith("SW-"): return 2
        return 9

    def weld_num(wid: str) -> int:
        m = re.search(r"-([0-9]+)", wid)
        return int(m.group(1)) if m else 10**9

    uniq.sort(key=lambda r: (family_rank(r["weld_id"]), weld_num(r["weld_id"]), r["page"]))
    return uniq

# ---------------- Optional LLM enrichment ----------------
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
            data = json.loads(resp.choices[0].message.content)
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

# ---------------- Excel export (safe for empty/NaN) ----------------

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Robust writer that never crashes on empty/NaN data and auto-sizes columns.
    """
    buf = io.BytesIO()
    safe_df = (df if df is not None else pd.DataFrame()).copy()
    if safe_df.empty:
        safe_df = pd.DataFrame(columns=["Weld Number", "Joint Size (ND)", "Joint Type", "Material Description", "Source Page"])
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        safe_df.to_excel(writer, index=False, sheet_name="Weld Log")
        ws = writer.sheets["Weld Log"]
        for i, col in enumerate(safe_df.columns):
            series = safe_df[col].astype(str)
            # handle all-NaN or empty safely
            mean_len = float(series.str.len().dropna().mean()) if len(series) else 0.0
            header_len = len(str(col))
            width = min(50, max(12, int(round(max(header_len, mean_len) + 6))))
            ws.set_column(i, i, width)
    buf.seek(0)
    return buf.getvalue()
