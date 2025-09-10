import os
import io
import json
import time
import re
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

APP_TITLE = "Weld Log Extractor — Deterministic Build"

def get_api_key() -> str:
    # 1) Prefer st.secrets if provided
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    # 2) Then session memory
    if not key:
        key = st.session_state.get("api_key", "")
    return key or ""

def set_api_key_in_session(k: str):
    st.session_state["api_key"] = k.strip()

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload ISO PDF → deterministic text parsing → structured weld log → Excel")

    with st.sidebar:
        st.subheader("Settings")
        api_key = st.text_input("OpenAI API Key", value=get_api_key(), type="password")
        remember = st.checkbox("Remember in session", value=True)
        if remember:
            set_api_key_in_session(api_key)

        st.markdown("**Extraction Controls**")
        exclude_field_welds = st.checkbox("Exclude Field Welds (FW)", value=True, help="Removes FW / FIELD WELD from results.")
        strict_regex = st.checkbox("Strict weld pattern (W-### only)", value=False, help="If ON, only accepts W-<number> forms.")
        llm_enrichment = st.checkbox("Use LLM to fill Size / Type / Material (temp=0)", value=False)
        st.caption("Tip: Leave LLM OFF first to verify the weld count is correct & stable. Then turn it ON to try fill details.")

    uploaded = st.file_uploader("Upload PDF isometric drawing", type=["pdf"])

    if uploaded is None:
        st.info("Upload an ISO PDF to begin.")
        return

    with st.spinner("Reading PDF text deterministically…"):
        pdf_bytes = uploaded.read()
        pages = extract_pdf_text_per_page(pdf_bytes)

    # Show quick debug on raw text length per page
    with st.expander("Debug view: Raw text per page"):
        for p in pages:
            st.markdown(f"**Page {p['page']}** — {len(p['text'])} chars")
            st.code(p["text"][:3000] or "<no text on this page>")

    with st.spinner("Finding weld candidates…"):
        candidates = find_weld_candidates(pages)

    with st.expander("Debug view: All candidates found"):
        st.write(candidates)

    with st.spinner("Filtering / normalizing…"):
        welds = filter_and_normalize_welds(
            candidates,
            exclude_field_welds=exclude_field_welds,
            strict_form=strict_regex
        )

    with st.expander("Debug view: Filtered weld IDs"):
        st.write(welds)

    # Build initial DataFrame
    df = pd.DataFrame({
        "Weld Number": [w["weld_id"] for w in welds],
        "Joint Size (ND)": ["" for _ in welds],
        "Joint Type": ["" for _ in welds],
        "Material Description": ["" for _ in welds],
        "Source Page": [w["page"] for w in welds],
        "Context": [w["context"] for w in welds],
    })

    if llm_enrichment:
        if not get_api_key():
            st.warning("LLM enrichment requires an OpenAI API key in the sidebar.")
        else:
            with st.spinner("LLM enrichment (deterministic, temp=0)…"):
                df = enrich_with_llm_fields(df, api_key=get_api_key())

    st.subheader("Weld Log (deterministic)")
    st.write(f"Total welds: **{len(df)}**")
    st.dataframe(df.drop(columns=["Context"]), use_container_width=True)

    # Download
    excel_bytes = df_to_excel_bytes(df.drop(columns=["Context"]))
    st.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name=f"weld_log_{int(time.time())}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.caption("If counts look off, open the Debug views above to see what was detected vs filtered.")

if __name__ == "__main__":
    main()
