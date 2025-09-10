import io
import re
import json
from typing import List, Dict, Any

import pdfplumber
import pandas as pd

# ---------- PDF TEXT EXTRACTION ----------

def extract_pdf_text_per_page(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Deterministic text extraction using pdfplumber (vector text → stable).
    Returns: [{"page": 1, "text": "..."}]
    """
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # extract_text() is deterministic given same file
            txt = page.extract_text() or ""
            # Normalize whitespace (deterministic)
            txt = "\n".join(line.strip() for line in txt.splitlines())
            pages.append({"page": i, "text": txt})
    return pages

# ---------- WELD ID DETECTION ----------

# Common patterns we accept (chosen for data-centre ISO style):
# - W-12, W-123, W 12, W:12, W_12 (normalized to W-12)
# - Sometimes "WELD-12" or "WELD 12" → normalized to W-12
# Exclusions:
# - FW, F/W, FIELD WELD
# - Leading text that implies field welds
WELD_PATTERN = re.compile(
    r"\b(?:(?:W(?:ELD)?)\s*[-_:]?\s*(\d{1,5}))\b",
    flags=re.IGNORECASE
)

FIELD_WELD_BLOCKERS = re.compile(
    r"\b(FW|F/W|FIELD\s*WELD|FIELDWELD)\b",
    flags=re.IGNORECASE
)

STRICT_W_FORM = re.compile(r"^W-\d{1,5}$")

def find_weld_candidates(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find raw candidates with context lines.
    Returns list of dicts: {weld_id_raw, page, context}
    """
    found = []
    for p in pages:
        text = p["text"]
        # Build context by lines
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            for m in WELD_PATTERN.finditer(line):
                num = m.group(1)
                raw = m.group(0)
                # Gather small context window
                ctx_lines = lines[max(0, idx-2): idx+3]
                ctx = " | ".join(ctx_lines)
                found.append({
                    "weld_id_raw": raw,
                    "number": num,
                    "page": p["page"],
                    "context": ctx
                })
    return found

def _normalize_weld_id(raw_num: str) -> str:
    # Normalize to "W-<number>"
    return f"W-{int(raw_num)}"

def filter_and_normalize_welds(candidates: List[Dict[str, Any]], exclude_field_welds: bool, strict_form: bool):
    """
    - Normalize to W-<n>
    - Exclude anything that looks like FW/Field Weld if exclude_field_welds=True
    - Stable unique by (weld_id, first occurrence page), keep earliest context
    - If strict_form=True: only keep things that match W-\d+ after normalization
    """
    rows = []
    for c in candidates:
        weld_id = _normalize_weld_id(c["number"])
        ctx = c["context"]

        # Exclude field welds by context
        if exclude_field_welds and FIELD_WELD_BLOCKERS.search(ctx):
            continue

        if strict_form and not STRICT_W_FORM.match(weld_id):
            continue

        rows.append({
            "weld_id": weld_id,
            "page": c["page"],
            "context": ctx,
        })

    # De-duplicate deterministically (first occurrence wins)
    seen = set()
    unique = []
    for r in rows:
        key = (r["weld_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    # Sort deterministically: by numeric value of weld, then by page
    def weld_num(wid: str) -> int:
        try:
            return int(wid.split("-")[1])
        except Exception:
            return 10**9

    unique.sort(key=lambda x: (weld_num(x["weld_id"]), x["page"]))
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
        "If size (ND), joint type, or material description are explicitly present near this weld, extract them."
        " Otherwise, leave them blank."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

# We use the official OpenAI Python SDK v1.x style if available.
def _openai_client(api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def enrich_with_llm_fields(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    client = _openai_client(api_key)
    if client is None:
        return df  # SDK not available; fail silently but deterministically

    # JSON schema for strict, consistent output
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
        messages = _build_llm_messages(str(row["Weld Number"]), str(row["Context"]))
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
                "Context": row["Context"],
            })
        except Exception:
            # Deterministic fallback: leave blanks if anything fails
            out_rows.append({
                "Weld Number": row["Weld Number"],
                "Joint Size (ND)": "",
                "Joint Type": "",
                "Material Description": "",
                "Source Page": row["Source Page"],
                "Context": row["Context"],
            })

    return pd.DataFrame(out_rows)

# ---------- EXPORT ----------

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Weld Log")
        # Auto column width
        worksheet = writer.sheets["Weld Log"]
        for i, col in enumerate(df.columns):
            width = min(50, max(12, int(df[col].astype(str).str.len().mean()) + 6))
            worksheet.set_column(i, i, width)
    buf.seek(0)
    return buf.getvalue()
