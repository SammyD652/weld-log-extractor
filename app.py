import os
import io
import base64
import json
from io import BytesIO
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
from openai import OpenAI

# ------------------------------
# Config & helpers
# ------------------------------
st.set_page_config(page_title="Weld Log Extractor (Auto-Zoom)", layout="wide")
st.title("🧠 Weld Log Extractor — Auto-Zoom Around SW Tags")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.info("Add your OpenAI key in **Settings → Secrets** as `OPENAI_API_KEY`, or set the env var on local.")
client = OpenAI(api_key=OPENAI_API_KEY)


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_pdf_to_images(pdf_bytes: bytes, zoom: float = 2.5) -> List[Image.Image]:
    """Render each PDF page to a sharp PIL image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


# ------------------------------
# GPT prompts (detector + reader)
# ------------------------------

SYSTEM_DETECT = (
    "You are a precise detector for piping isometric drawings. "
    "Given a page image, find EVERY weld tag bubble like 'SW 001', 'SW 017', etc. "
    "Return **ONLY JSON** with an array 'weld_tags'. "
    "Each item: {weld_number:int, box:[x,y,w,h], context_box:[x,y,w,h]}.\n"
    "- 'box' must be a tight rectangle around the hexagon SW bubble.\n"
    "- 'context_box' must be a larger rectangle (~3× the weld tag area) centered on the weld so it includes "
    "  nearby **square BOM refs**, reducers/tees/valves, and the local pipe segment.\n"
    "Do not include any other keys."
)

SYSTEM_READ = (
    "You are a weld size resolver. You will receive a cropped image around exactly one SW weld bubble.\n"
    "Apply these rules STRICTLY:\n"
    "1) Only trust numbers inside **square boxes** as BOM references. Ignore numbers without square boxes (spools, lengths).\n"
    "2) If the nearby BOM refs are the same ND, the weld is that ND.\n"
    "3) If refs differ, use geometry:\n"
    "   • Reducer (e.g., 65x25): big face = larger ND; small face = smaller ND.\n"
    "   • Reducing tee (e.g., 25x20): straight-through = large ND, branch outlet = small ND.\n"
    "4) Field vs Shop does NOT imply size.\n"
    "Return **ONLY JSON**: {weld_number:int, near_refs:[int], inferred_nd:int|null, notes:str}.\n"
    "If ambiguous, set inferred_nd=null and explain briefly in 'notes'."
)

USER_READ = (
    "Identify the nearby square-box BOM references and infer the weld ND (mm) using the rules. "
    "Return JSON only."
)


def gpt_detect_welds(page_img: Image.Image) -> List[Dict]:
    """Ask GPT-4o to locate SW tags and give context crops."""
    b64 = img_to_b64(page_img)
    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": SYSTEM_DETECT},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Detect all SW weld tags and return JSON only."},
                {"type": "input_image", "image_data": b64},
            ]}
        ],
        temperature=0
    )

    text = ""
    for blk in resp.output[0].content:
        if blk.type == "output_text":
            text += blk.text

    # Clean JSON if fenced
    clean = text.strip().strip("```").strip()
    if clean.lower().startswith("json"):
        clean = clean[4:].strip()

    data = json.loads(clean)
    return data.get("weld_tags", [])


def gpt_read_weld(crop_img: Image.Image, weld_number: int) -> Dict:
    """Ask GPT-4o to infer ND from the context crop."""
    b64 = img_to_b64(crop_img)
    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": SYSTEM_READ},
            {"role": "user", "content": [
                {"type": "input_text", "text": json.dumps({"weld_number": weld_number})},
                {"type": "input_text", "text": USER_READ},
                {"type": "input_image", "image_data": b64},
            ]}
        ],
        temperature=0
    )
    text = ""
    for blk in resp.output[0].content:
        if blk.type == "output_text":
            text += blk.text

    clean = text.strip().strip("```").strip()
    if clean.lower().startswith("json"):
        clean = clean[4:].strip()
    data = json.loads(clean)
    data["weld_number"] = weld_number
    return data


def clamp_box(box: List[int], W: int, H: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def crop(img: Image.Image, box: List[int]) -> Image.Image:
    x, y, w, h = clamp_box(box, *img.size)
    return img.crop((x, y, x + w, y + h))


# ------------------------------
# UI
# ------------------------------
with st.sidebar:
    st.subheader("Options")
    show_crops = st.checkbox("Show auto-zoom crops", value=True)
    st.caption("Useful for quick QA before export.")
    strict_ambig = st.checkbox("Mark ambiguous (don’t guess)", value=True)
    st.caption("If on, ambiguous welds will have ND = blank and a note.")

uploaded = st.file_uploader("📄 Upload an isometric PDF", type=["pdf"])

if uploaded:
    pdf_bytes = uploaded.read()
    pages = render_pdf_to_images(pdf_bytes, zoom=2.5)

    all_rows = []
    crop_gallery = []

    prog = st.progress(0.0, text="Scanning pages for SW tags…")
    total = len(pages)
    for i, page_img in enumerate(pages, start=1):
        prog.progress(i / total, text=f"Detecting welds on page {i}/{total}…")
        detections = gpt_detect_welds(page_img)

        # Deduplicate if detector finds same weld twice; prefer first
        seen = set()
        clean_dets = []
        for d in detections:
            wn = int(d["weld_number"])
            if wn in seen:
                continue
            seen.add(wn)
            clean_dets.append(d)

        for det in clean_dets:
            wn = int(det["weld_number"])
            ctx = det.get("context_box") or det["box"]
            ctx_crop = crop(page_img, ctx)
            read = gpt_read_weld(ctx_crop, wn)

            nd = read.get("inferred_nd")
            notes = read.get("notes", "")
            if strict_ambig and (nd is None):
                # leave as blank; force review
                pass

            all_rows.append({
                "Weld Number": wn,
                "Shop/Field": "Shop",  # these drawings use SW; adjust if FW present
                "Weld Size (ND)": nd,
                "BOM Refs Near": ",".join(map(str, read.get("near_refs", []))),
                "Notes": notes,
                "Page": i
            })

            if show_crops:
                crop_gallery.append((wn, ctx_crop.copy()))

    if not all_rows:
        st.warning("No weld tags detected. If your PDF is very light/complex, try re-exporting as a vector-rich PDF.")
        st.stop()

    df = pd.DataFrame(all_rows).sort_values(["Page", "Weld Number"]).reset_index(drop=True)
    st.subheader("📋 Weld Log (AI sized from auto-zoom crops)")
    st.dataframe(df, use_container_width=True)

    # Download Excel
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="Weld Log")
    st.download_button(
        "⬇️ Download Excel",
        data=out.getvalue(),
        file_name=f"{uploaded.name.rsplit('.',1)[0]}_weld_log.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Optional crop gallery
    if show_crops:
        st.subheader("🔎 Auto-zoom crops (for QA)")
        cols = st.columns(4)
        for idx, (wn, im) in enumerate(sorted(crop_gallery, key=lambda t: t[0])):
            with cols[idx % 4]:
                st.image(im, caption=f"SW {wn}", use_container_width=True)

else:
    st.caption("Upload a full isometric PDF. The app will auto-zoom around each SW tag and size the weld using BOM refs.")
