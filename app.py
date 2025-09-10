import time
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
    st.caption("Upload ISO PDF → deterministic text parsing (+OCR fallback) → weld IDs → Excel. Focus: accuracy & consistency.")

    with st.sidebar:
        st.subheader("Settings")
        api_key = st.text_input("OpenAI API Key", value=get_api_key(), type="password")
        remember = st.checkbox("Remember in session", value=True)
        if remember:
            set_api_key_in_session(api_key)

        st.markdown("**Extraction Controls**")
        exclude_field_welds = st.checkbox(
            "Exclude Field Welds (FW)", value=True,
            help="Remove FW / FIELD WELD by context."
        )
        strict_regex = st.checkbox(
            "Strict pattern (W-### / SW-### / BW-### only)", value=False,
            help="If ON, keep only normalized forms."
        )
        force_ocr = st.checkbox(
            "Force OCR fallback (for scanned PDFs)", value=True,
            help="Turn ON if Raw text has little/no content."
        )
        aggressive = st.checkbox(
            "Aggressive SW/BW/W finder (join ladders, fix O↔0, l↔1)", value=True
        )
        llm_enrichment = st.checkbox(
            "Use LLM to fill Size / Type / Material (temp=0)", value=False
        )
        st.caption("Tip: First verify the weld COUNT with LLM OFF. Then try LLM to fill details.")

    uploaded = st.file_uploader("Upload PDF isometric drawing", type=["pdf"])
    if uploaded is None:
        st.info("Upload an ISO PDF to begin.")
        return
    pdf_bytes = uploaded.read()

    with st.spinner("Reading PDF text…"):
        pages = extract_pdf_text_per_page(pdf_bytes, enable_ocr_fallback=force_ocr)

    with st.expander("Debug view: Raw text per page"):
        for p in pages:
            preview = p["text"] if p["text"] else "<no text on this page>"
            st.markdown(f"**Page {p['page']}** — {len(p['text'])} chars")
            st.code(preview[:3000])

    with st.spinner("Finding weld candidates…"):
        candidates, match_preview = find_weld_candidates(pages, aggressive=aggressive, return_preview=True)

    with st.expander("Debug view: Matches preview (strings we matched)"):
        st.write(match_preview[:200])

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
    st.dataframe(df.drop(columns=["Context"], errors="ignore"), use_container_width=True)

    excel_df = df.drop(columns=["Context"], errors="ignore")
    excel_bytes = df_to_excel_bytes(excel_df)
    st.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name=f"weld_log_{int(time.time())}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if len(df) == 0:
        st.info("Still 0? Keep **Force OCR fallback** and **Aggressive finder** ON. Make sure **Strict pattern** is OFF for the first pass.")

if __name__ == "__main__":
    main()
