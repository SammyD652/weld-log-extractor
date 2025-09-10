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
    # keep simple; secrets optional
    return st.session_state.get("api_key", "")

def set_api_key_in_session(k: str):
    st.session_state["api_key"] = (k or "").strip()

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload ISO PDF → deterministic text parsing (+OCR fallback) → weld IDs → Excel. Focus: accuracy & consistency.")

    # ---------- Sidebar controls ----------
    with st.sidebar:
        st.subheader("Settings")
        api_key = st.text_input("OpenAI API Key", value=get_api_key(), type="password")
        if st.checkbox("Remember in session", value=True):
            set_api_key_in_session(api_key)

        st.markdown("**Extraction Controls**")
        exclude_field_welds = st.checkbox("Exclude Field Welds (FW)", value=True)
        strict_regex = st.checkbox("Strict pattern (W-### / SW-### / BW-### only)", value=False)
        force_ocr = st.checkbox("Force OCR fallback (for scanned PDFs)", value=True)
        aggressive = st.checkbox("Aggressive SW/BW/W finder (join ladders, fix OCR)", value=True)
        llm_enrichment = st.checkbox("Use LLM to fill Size / Type / Material (temp=0)", value=False)
        st.caption("Tip: first get the **weld count** correct with LLM OFF; then try LLM to fill details.")

    # ---------- Upload ----------
    uploaded = st.file_uploader("Upload PDF isometric drawing", type=["pdf"])
    if not uploaded:
        st.info("Upload a PDF to begin.")
        return
    pdf_bytes = uploaded.read()

    # ---------- Parse text ----------
    with st.spinner("Reading PDF text…"):
        pages = extract_pdf_text_per_page(pdf_bytes, enable_ocr_fallback=force_ocr)

    with st.expander("Debug view: Raw text per page"):
        for p in pages:
            st.markdown(f"**Page {p['page']}** — {len(p['text'])} chars")
            st.code((p["text"] or "")[:2000])

    # ---------- Find candidates ----------
    with st.spinner("Finding weld candidates…"):
        candidates, match_preview = find_weld_candidates(
            pages, aggressive=aggressive, return_preview=True
        )

    with st.expander("Debug view: Matches preview (strings we matched)"):
        st.write(match_preview[:200])

    with st.expander("Debug view: All candidates found"):
        st.write(candidates)

    # ---------- Filter / normalize ----------
    welds = filter_and_normalize_welds(
        candidates,
        exclude_field_welds=exclude_field_welds,
        strict_form=strict_regex,
    )
    with st.expander("Debug view: Filtered weld IDs"):
        st.write(welds)

    # ---------- Build DF ----------
    df = pd.DataFrame({
        "Weld Number": [w["weld_id"] for w in welds],
        "Joint Size (ND)": ["" for _ in welds],
        "Joint Type": ["" for _ in welds],
        "Material Description": ["" for _ in welds],
        "Source Page": [w["page"] for w in welds],
        "Context": [w["context"] for w in welds],
    })

    # ---------- Optional LLM enrichment ----------
    if llm_enrichment and api_key:
        with st.spinner("LLM enrichment (temp=0)…"):
            df = enrich_with_llm_fields(df, api_key=api_key)

    # ---------- Table + Download ----------
    st.subheader("Weld Log (deterministic)")
    st.write(f"Total welds: **{len(df)}**")
    st.dataframe(df.drop(columns=["Context"], errors="ignore"), width="stretch")

    excel_bytes = df_to_excel_bytes(df.drop(columns=["Context"], errors="ignore"))
    st.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name=f"weld_log_{int(time.time())}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if len(df) == 0:
        st.info("Still 0? Keep **Force OCR fallback** + **Aggressive finder** ON and **Strict pattern** OFF for the first pass.")

if __name__ == "__main__":
    main()
