import io
import re
from typing import List, Tuple, Dict

import streamlit as st
import pandas as pd

from extractor.pdf_text import extract_text_per_page
from extractor.pdf_ocr import ocr_pages_needing_help
from extractor.parse_iso import parse_weld_data_from_pages
from extractor.llm_validate import validate_with_llm_if_enabled
from extractor.schema import WeldRecord
from utils.io import to_excel_bytes, to_csv_bytes

APP_TITLE = "Weld Log Extraction (Online Only)"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Sidebar settings
st.sidebar.title("Settings")
st.sidebar.caption("This app runs fully online on Streamlit Cloud.")

st.sidebar.subheader("Validation (optional)")
use_llm = st.sidebar.toggle(
    "Use OpenAI Vision to validate tricky rows (optional)", value=False
)
st.sidebar.caption(
    "Keep off to avoid cost. Turn on only if results look off and you have a key in Secrets."
)

st.sidebar.subheader("OCR Options")
dpi = st.sidebar.slider(
    "OCR render DPI (pdf2image)", 150, 300, 200, step=25
)
psm = st.sidebar.selectbox(
    "Tesseract PSM (page segmentation mode)",
    options=[6, 4, 3],
    index=0,
    help="6 = Assume a single uniform block of text, 4 = single column, 3 = fully automatic",
)

st.sidebar.divider()
st.sidebar.subheader("Logs")
show_raw_text = st.sidebar.toggle("Show extracted text (debug)", False)
show_ocr_snippets = st.sidebar.toggle("Show OCR snippets (debug)", False)

# Main interface
st.title(APP_TITLE)
st.write(
    "Upload full **isometric PDF drawings**. The app will try the PDF text layer first,"
    " then OCR only the pages that need it. It parses **Weld Number, Joint Size (ND/DN),"
    " Joint Type, Material Description**, shows a preview, and lets you download **Excel/CSV**."
)

# Progress panel
progress_box = st.container()
with progress_box:
    st.markdown("### Progress")
    progress_status = st.empty()

# File uploader
uploaded_files = st.file_uploader(
    "Upload one or more PDF isometric drawings", type=["pdf"], accept_multiple_files=True
)

# Helper to check if an OpenAI key exists in secrets
def has_llm_key() -> bool:
    try:
        return bool(st.secrets.get("OPENAI_API_KEY", ""))
    except Exception:
        return False

if uploaded_files:
    # Step 1: Read PDFs (text layer)
    progress_status.info("Step 1/5: Reading PDFs (text layer)…")
    pdf_buffers: List[Tuple[str, bytes]] = [(uf.name, uf.read()) for uf in uploaded_files]

    # Extract text per page for each file
    all_docs_pages = []  # list of dict: {"file": name, "page_index": i, "text": "..."}
    for name, data in pdf_buffers:
        text_pages = extract_text_per_page(io.BytesIO(data))
        for i, t in enumerate(text_pages):
            all_docs_pages.append({"file": name, "page_index": i, "text": t})

    # Step 2: Decide which pages need OCR (no/low text)
    progress_status.info("Step 2/5: Checking pages that need OCR…")
    need_ocr_mask = []
    for page in all_docs_pages:
        text = (page["text"] or "").strip()
        need_ocr_mask.append(len(text) < 200)

    pages_needing_ocr = [
        (p["file"], p["page_index"]) for p, need in zip(all_docs_pages, need_ocr_mask) if need
    ]
    ocr_report = {}

    # Step 3: OCR only pages that need it
    if pages_needing_ocr:
        progress_status.info(f"Step 3/5: OCR {len(pages_needing_ocr)} page(s)…")
        by_file: Dict[str, List[int]] = {}
        for fname, pidx in pages_needing_ocr:
            by_file.setdefault(fname, []).append(pidx)
        for (name, data) in pdf_buffers:
            if name in by_file:
                ocr_texts, per_page_logs = ocr_pages_needing_help(
                    io.BytesIO(data), want_pages=by_file[name], dpi=dpi, psm=psm
                )
                for page_idx, txt in ocr_texts.items():
                    idx = next(
                        i
                        for i, p in enumerate(all_docs_pages)
                        if p["file"] == name and p["page_index"] == page_idx
                    )
                    all_docs_pages[idx]["text"] = (all_docs_pages[idx]["text"] or "") + "\n" + txt
                ocr_report[name] = per_page_logs
    else:
        progress_status.info("Step 3/5: No OCR needed — all pages had a text layer.")

    # Debug panels
    with st.expander("Text Extraction / OCR Logs", expanded=False):
        st.write("Pages extracted:", len(all_docs_pages))
        if pages_needing_ocr:
            st.write("Pages OCR’d:", pages_needing_ocr)
        if show_raw_text:
            for p in all_docs_pages:
                st.markdown(f"**{p['file']} — Page {p['page_index']+1}**")
                st.code((p["text"] or "")[:5000])
        if show_ocr_snippets and ocr_report:
            st.write(ocr_report)

    # Step 4: Parse structured weld data
    progress_status.info("Step 4/5: Parsing weld data…")
    parsed_records = parse_weld_data_from_pages(all_docs_pages)

    valid_records: List[WeldRecord] = []
    invalid_rows = []
    for rec in parsed_records:
        try:
            vr = WeldRecord(**rec)
            valid_records.append(vr)
        except Exception as e:
            row = rec.copy()
            row["_validation_error"] = str(e)
            invalid_rows.append(row)

    # Optional LLM validation
    if use_llm:
        if not has_llm_key():
            st.warning("OpenAI key not found in Streamlit Secrets. Skipping LLM validation.")
        else:
            progress_status.info("Step 4.5/5: Validating with OpenAI Vision…")
            valid_records = validate_with_llm_if_enabled(valid_records, all_docs_pages)

    # Deduplicate by (weld_number, joint_size, joint_type)
    seen = set()
    deduped = []
    for vr in valid_records:
        key = (
            (vr.weld_number or "").upper().strip(),
            (vr.joint_size or "").upper().strip(),
            (vr.joint_type or "").upper().strip(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(vr)

    # Step 5: Preview & Export
    progress_status.success("Step 5/5: Export ready ✅")
    st.markdown("### Preview")
    if deduped:
        df = pd.DataFrame([vr.model_dump() for vr in deduped])
        st.dataframe(df, use_container_width=True, hide_index=True)

        xlsx_bytes = to_excel_bytes(df, sheet_name="Weld Log")
        csv_bytes = to_csv_bytes(df)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Excel (.xlsx)",
                data=xlsx_bytes,
                file_name="weld_log.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col2:
            st.download_button(
                "⬇️ Download CSV (.csv)",
                data=csv_bytes,
                file_name="weld_log.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "No valid rows parsed yet. Try enabling OCR (already automatic for low-text pages), adjust DPI,"
            " or enable LLM validation in the sidebar."
        )

    if invalid_rows:
        with st.expander("Rows that failed validation (dropped)", expanded=False):
            st.dataframe(pd.DataFrame(invalid_rows), use_container_width=True, hide_index=True)
else:
    st.info("👆 Upload one or more **PDF** files to begin.")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
**Pipeline**
1) Try PDF text layer (fast).
2) For pages with little or no text, render to images → OCR with Tesseract.
3) Parse heuristics for Weld Number (W-###/W###/WN-###), ND/DN sizes, joint type codes (BW/SW/THD/SO/etc.), and material description.
4) Validate with Pydantic; optional OpenAI Vision cross-check.
5) Preview, then download Excel/CSV.
"""
    )
