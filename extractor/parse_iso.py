"""
Module to parse isometric drawing text for weld log data.

We apply simple heuristics to identify weld numbers, joint sizes (ND/DN),
joint types (e.g., BW, SW, THD), and material descriptions from BOM/spec
sections of the PDF text.

This module is intentionally heuristic: it starts with general patterns that
should capture the majority of cases. After testing on real drawings, you
should refine the patterns to match your company's drawing conventions.
"""

import re
from typing import List, Dict, Any


BOM_HEADERS_HINTS = [
    "bill of materials",
    "bom",
    "materials",
    "specification",
    "spec table",
    "material list",
]

# Patterns for identifying weld numbers (W-###, W123, WN-###, WELD-###)
WELD_NUMBER_PATTERNS = [
    r"\bW[-\s]?(\d{1,5})\b",
    r"\bWN[-\s]?(\d{1,5})\b",
    r"\bWELD[-\s]?(\d{1,5})\b",
]

# Patterns for ND/DN/Ø sizes like ND 25, DN25, Ø25, ND25. Allow decimals.
SIZE_PATTERNS = [
    r"\bN[DN]\s?(\d{1,3}(?:\.\d+)?)\b",
    r"\bDN\s?(\d{1,3}(?:\.\d+)?)\b",
    r"[Ø⌀]\s?(\d{1,3}(?:\.\d+)?)",
    r"\bND\s?(\d{1,3}(?:\.\d+)?)\b",
]

# Joint type codes we expect to see
JOINT_TYPES = [
    "BW",
    "BUTT",
    "SW",
    "SO",
    "THD",
    "THRD",
    "FL",
    "FLG",
    "FLANGE",
    "SOCKET",
    "THREADED",
]


def _find_all(patterns, text):
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            yield m


def _closest_bom_text(page_texts: List[str]) -> str:
    """Return concatenated lines that likely form a BOM/spec area."""
    out_lines: List[str] = []
    for t in page_texts:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            low = ln.lower()
            if any(h in low for h in BOM_HEADERS_HINTS):
                window = lines[max(0, i - 10) : i + 60]
                out_lines.extend(window)
    return "\n".join(out_lines[:8000])


def parse_weld_data_from_pages(pages: List[dict]) -> List[Dict[str, Any]]:
    """
    Parse a list of pages (dicts with keys 'file', 'page_index', 'text') and
    return a list of potential weld log rows.

    Returns rows with keys: weld_number, joint_size, joint_type, material_description,
    source_file, source_page.
    """
    by_file: Dict[str, List[str]] = {}
    for p in pages:
        by_file.setdefault(p["file"], []).append(p.get("text") or "")

    bom_snippets = {f: _closest_bom_text(txts) for f, txts in by_file.items()}

    rows: List[Dict[str, Any]] = []
    for p in pages:
        raw = p.get("text") or ""
        # 1) find weld numbers on the page
        weld_nums: List[str] = []
        for m in _find_all(WELD_NUMBER_PATTERNS, raw):
            val = m.group(1)
            weld_nums.append(f"W-{val}")

        # 2) find ND/DN sizes on the page
        sizes = [m.group(1) for m in _find_all(SIZE_PATTERNS, raw)]
        size_pick = sizes[0] if sizes else None

        # 3) joint type: pick the first occurrence of any code
        jt = None
        for code in JOINT_TYPES:
            pat = r"\b" + re.escape(code) + r"\b"
            if re.search(pat, raw, flags=re.IGNORECASE):
                jt = code.upper()
                break

        # 4) material description: heuristically pick a line from BOM/snippet
        mat_desc = None
        if bom_snippets.get(p["file"]):
            lines = [ln.strip() for ln in bom_snippets[p["file"]].splitlines() if ln.strip()]
            picked = None
            if size_pick:
                size_token = rf"(?:ND|DN|Ø|⌀)\s?{re.escape(size_pick)}\b"
                for ln in lines:
                    if re.search(size_token, ln, flags=re.IGNORECASE):
                        picked = ln
                        break
            if not picked:
                for ln in lines:
                    if re.search(r"\b(pipe|carbon|stainless|cs|ss|sch|schedule|seamless|astm|a\d+)\b", ln, flags=re.IGNORECASE):
                        picked = ln
                        break
            if not picked and lines:
                picked = lines[0]
            mat_desc = picked

        for w in weld_nums:
            rows.append(
                {
                    "weld_number": w,
                    "joint_size": f"ND {size_pick}" if size_pick else None,
                    "joint_type": jt,
                    "material_description": mat_desc,
                    "source_file": p["file"],
                    "source_page": p["page_index"] + 1,
                }
            )

    # If no rows, salvage ND size-only rows
    if not rows:
        for p in pages:
            raw = p.get("text") or ""
            sizes = [m.group(1) for m in _find_all(SIZE_PATTERNS, raw)]
            for s in sizes:
                rows.append(
                    {
                        "weld_number": None,
                        "joint_size": f"ND {s}",
                        "joint_type": None,
                        "material_description": None,
                        "source_file": p["file"],
                        "source_page": p["page_index"] + 1,
                    }
                )

    return rows
