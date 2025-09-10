import os
import io
import json
import time
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

from parsers import (
    extract_pdf_text_per_page,
    find_weld_candidates,
    filter_and_normalize_welds,
    enrich_with_llm_fields,
    df_to_excel_bytes,
)

APP_TITLE = "Weld Log Extractor — Deterministic + OCR Fallback"

def get_api_key() -> str:
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        key = st.session_state.get("api_key", "")
    return key or ""

def set_api_key_in_session(k: str):
    st.session_state["api_key"] = (k or "").strip()

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload ISO PDF → deterministic text parsing (+OCR fallback for scans) → structured weld log → Excel")

    with st.sidebar:
        st.subheader("Settings")
        api_key = st.text_input("OpenAI API Key", value=get_api_key(), type="password")
        remember = st.checkbox("Remember in session", value=True)
        if remember:
            set_api_key_in_session(api_key)

        st.markdown("**Extraction Controls**")
        exclude_field_welds = st.checkbox("Exclude Field Welds (FW)", value=True,
                                           help="Removes FW / FIELD WELD mentions by context.")
        strict_regex = st.checkbox("Strict weld pattern (W-### only)", value=False,
                                   help="If ON, only accepts W-<number> forms.")
        force_ocr = st.checkbox("Force OCR fallback (for scanned PDFs)", value=False,
                                help="Turn ON if Raw text view shows little/no text.")
        llm_enrichment = st.checkbox("Use LLM to fill Size / Type / Material (temp=0)", value=False)
        st.caption("Tip: First verify the weld COUNT with LLM OFF. Then try LLM to fill details.")

    uploaded = st.file_uploader("Upload PDF isometric drawing", type=["pdf"])

    if uploaded is None:
        st.info("Upload an ISO PDF to begin.")
        return

    pdf_bytes = uploaded.read()

    with st.spinner("Reading PDF text deterministically…"):
        pages = extract_pdf_text_per_page(pdf_bytes, enable_ocr_fallback=force_ocr)

    # Debug: show raw page text
    with st.expander("Debug view: Raw text per page"):
        for p in pages:
            preview = p["text"] if p["text"] else "<no text on this page>"
            st.markdown(f"**Page {p['page']}** — {len(p['text'])} chars")
            st.code(preview[:3000])

    # Find candidates
    with st.spinner("Finding weld candidates…"):
        candidates = find_weld_candidates(pages)

    with st.expander("Debug view: All candidates found"):
        st.write(candidates)

    # Filter / normalize
    with st.spinner("Filtering / normalizing…"):
        welds = filter_and_normalize_welds(
            candidates,
            exclude_field_welds=exclude_field_welds,
            strict_form=strict_regex
        )

    with st.expander("Debug view: Filtered weld IDs"):
        st.write(welds)

    # Build DataFrame
    df = pd.DataFrame({
        "Weld Number": [w["weld_id"] for w in welds],
        "Joint Size (ND)": ["" for _ in welds],
        "Joint Type": ["" for _ in welds],
        "Material Description": ["" for _ in welds],
        "Source Page": [w["page"] for w in welds],
        "Context": [w["context"] for w in welds],
    })

    # Optional LLM enrichment
    if llm_enrichment:
        if not get_api_key():
            st.warning("LLM enrichment requires an OpenAI API key in the sidebar.")
        else:
            with st.spinner("LLM enrichment (deterministic, temp=0)…"):
                df = enrich_with_llm_fields(df, api_key=get_api_key())

    st.subheader("Weld Log (deterministic)")
    st.write(f"Total welds: **{len(df)}**")

    # Show table (new API: width='stretch' instead of deprecated use_container_width)
    st.dataframe(df.drop(columns=["Context"], errors="ignore"), width='stretch')

    # Safe Excel export (works even if df is empty)
    excel_df = df.drop(columns=["Context"], errors="ignore")
    excel_bytes = df_to_excel_bytes(excel_df)
    st.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name=f"weld_log_{int(time.time())}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=excel_df is None
    )

    if len(df) == 0:
        st.info("No weld IDs detected. If your PDF is scanned/image-only, toggle **Force OCR fallback** in the sidebar. "
                "Also try turning **Strict pattern** OFF to allow forms like `W 12`, `W_12`, `W:12`.")

if __name__ == "__main__":
    main()
