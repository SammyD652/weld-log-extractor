import io, re, json
from typing import List, Dict, Any, Tuple
import pdfplumber, pandas as pd

try:
    import pypdfium2 as pdfium
    import pytesseract
    from PIL import Image, ImageOps, ImageFilter
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------------- PDF Extraction ----------------

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
    img = page.render(scale=dpi/72).to_pil().convert("L")
    img = ImageOps.autocontrast(img).filter(ImageFilter.SHARPEN)
    config = "-l eng --oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_:#"
    return pytesseract.image_to_string(img, config=config)

def extract_pdf_text_per_page(pdf_bytes: bytes, enable_ocr_fallback: bool = False) -> List[Dict[str, Any]]:
    pages = _page_text_pdfplumber(pdf_bytes)
    out = []
    for p in pages:
        text = p["text"]
        if enable_ocr_fallback or len(text.strip()) < 30:
            ocr_text = _ocr_page(pdf_bytes, p["page"]-1)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        out.append({"page": p["page"], "text": text})
    return out

# ---------------- Weld ID Detection ----------------

BASE = re.compile(r"\b(?:(SW|BW|W))(?:ELD)?[-_:\s]*(?:No\.|#)?\s*([0-9OIl]{1,5}[A-Z]?)\b", re.I)
STRICT_W_FORM = re.compile(r"^(?:W|SW|BW)-\d{1,5}[A-Z]?$", re.I)
FIELD_WELD_BLOCKERS = re.compile(r"\b(FW|F/W|FIELD\s*WELD)\b", re.I)

def _fix_digits(s: str) -> str:
    return s.replace("O","0").replace("o","0").replace("I","1").replace("l","1")

def _normalize(prefix: str, raw: str) -> str:
    raw = _fix_digits(raw).upper()
    m = re.match(r"(\d{1,5})([A-Z]?)$", raw)
    if m: return f"{prefix}-{int(m.group(1))}{m.group(2)}"
    return f"{prefix}-{raw}"

def _stitch(lines: List[str]) -> List[str]:
    stitched, buf = [], []
    def flush():
        if buf: stitched.append("".join(buf)); buf.clear()
    for ln in lines:
        if len(ln.strip())==1 and ln.strip().isalnum(): buf.append(ln.strip())
        else: flush(); stitched.append(ln)
    flush()
    return stitched

def _scan_windows(lines: List[str], idx: int, max_lines=10) -> List[str]:
    out=[]
    for k in range(1,max_lines+1):
        if idx+k<=len(lines):
            seg=" ".join(lines[idx:idx+k])
            out.append(re.sub(r"\s+"," ",seg))
            out.append(re.sub(r"[^A-Za-z0-9:_#-]+","",seg))
    return out

def find_weld_candidates(pages: List[Dict[str, Any]], aggressive=True, return_preview=False) -> Tuple[List[Dict[str, Any]], List[str]]:
    found, preview=[],[]
    for p in pages:
        lines=_stitch((p["text"] or "").split("\n"))
        for idx in range(len(lines)):
            scan=_scan_windows(lines, idx, 12 if aggressive else 1)
            for w in scan:
                for m in BASE.finditer(w):
                    pref,num=(m.group(1) or "W").upper(), m.group(2)
                    ctx=" | ".join(lines[max(0,idx-3):idx+6])
                    found.append({"prefix":pref,"number":num,"page":p["page"],"context":ctx})
                    preview.append(w)
    return (found,preview) if return_preview else (found,[])

def filter_and_normalize_welds(cands: List[Dict[str,Any]], exclude_fw: bool, strict: bool):
    rows=[]
    for c in cands:
        wid=_normalize(c["prefix"], c["number"])
        if exclude_fw and FIELD_WELD_BLOCKERS.search(c["context"]): continue
        if strict and not STRICT_W_FORM.match(wid): continue
        rows.append({"weld_id":wid,"page":c["page"],"context":c["context"]})
    seen=set(); out=[]
    for r in rows:
        if r["weld_id"] not in seen:
            seen.add(r["weld_id"]); out.append(r)
    return out

# ---------------- Export ----------------

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf=io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer,index=False,sheet_name="Weld Log")
    buf.seek(0); return buf.getvalue()
