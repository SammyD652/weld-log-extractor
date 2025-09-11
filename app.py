import io
import base64
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
from openai import OpenAI

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Weld Log Extractor — Auto-Zoom Around SW Tags", layout="centered")

st.title("🧠 Weld Log Extractor — Auto-Zoom Around SW Tags")

with st.sidebar:
    show_crops = st.toggle("Show auto-zoom crops", value=True, help="Useful for quick QA before export.")
    mark_ambiguous = st.toggle("Mark ambiguous (don't guess)", value=True, help="If on, ambiguous welds will have ND = blank and a note.")

api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")
uploaded_pdf = st.file_uploader("📄 Upload an isometric PDF", type=["pdf"])

run = st.button("Detect welds")

# ---------------------------
# Helpers
# ---------------------------

def img_to_b64(img: Image.Image) -> str:
    """Return base64 (no prefix) PNG for PIL image."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def page_images_from_pdf(file_bytes: bytes, dpi: int = 220) -> List[Image.Image]:
    """
    Render each page of the PDF to a PIL Image using PyMuPDF.
    dpi ~ 220 gives good OCR without huge payloads.
    """
    imgs: List[Image.Image] = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # scale from 72 dpi
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imgs.append(img)
    doc.close()
    return imgs

def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def response_text(resp: Any) -> str:
    """
    Extract text safely from Responses API object across SDK minor changes.
    """
    # Prefer .output_text when available
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # Fallback: stitch any output_text fragments
    try:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "summary_text"):
                    txt = getattr(c, "text", None)
                    if txt:
                        chunks.append(txt)
        if chunks:
            return "\n".join(chunks)
    except Exception:
        pass

    # Last resort: string-ify
    return str(resp)

# ---------------------------
# GPT calls
# ---------------------------

SYSTEM_DETECT = (
    "You are a vision assistant that reads isometric piping drawings. "
    "Task: Find all weld tags (e.g., 'SW-015', 'SW 15', 'W15', etc.). "
    "Return a JSON array of objects with: weld_number (integer), "
    "x, y (pixel center on the page image), and an explanation if ambiguous. "
    "Do NOT guess numbers you cannot read—mark them as ambiguous."
)

def gpt_detect_welds(client: OpenAI, page_img: Image.Image, mark_ambiguous: bool = True) -> List[Dict]:
    """
    Ask GPT-4o to locate SW tags on the page and return JSON detections.
    Uses the Responses API with correct multimodal input types.
    """
    b64 = img_to_b64(page_img)
    data_url = f"data:image/png;base64,{b64}"

    user_prompt = (
        "Detect all weld number tags on this page image. "
        "Output STRICT JSON ONLY (no markdown, no prose): "
        '[{"weld_number": <int or null>, "x": <int>, "y": <int>, "ambiguous": <true|false>, "note": "<string>"}]. '
        "If unreadable, set weld_number = null, ambiguous = true, and add a short note. "
    )
    if not mark_ambiguous:
        user_prompt += "If unsure, make your best estimate but add note='low confidence'."

    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_DETECT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": {"url": data_url}},
                ],
            },
        ],
        response_format={"type": "json_object"}  # we will still parse the text
    )

    txt = response_text(resp)
    # The model may return either pure array or object-wrapped; handle both
    import json
    detections: List[Dict] = []
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            detections = data
        elif isinstance(data, dict):
            # accept {"detections":[...]} or {"data":[...]} or {"result":[...]}
            for key in ("detections", "data", "result", "items"):
                if key in data and isinstance(data[key], list):
                    detections = data[key]
                    break
    except Exception:
        # If parsing fails, return empty to avoid crashing the UI
        detections = []

    # Normalize/clean
    cleaned: List[Dict] = []
    for d in detections:
        try:
            wn = d.get("weld_number")
            if wn is not None:
                try:
                    wn = int(wn)
                except Exception:
                    wn = None
            cleaned.append({
                "weld_number": wn,
                "x": int(d.get("x", 0)),
                "y": int(d.get("y", 0)),
                "ambiguous": bool(d.get("ambiguous", wn is None)),
                "note": str(d.get("note", "")).strip()
            })
        except Exception:
            continue
    return cleaned

# ---------------------------
# Main run
# ---------------------------

if run:
    if not api_key:
        st.error("Please enter your OpenAI API key.")
        st.stop()
    if not uploaded_pdf:
        st.error("Please upload a PDF.")
        st.stop()

    client = get_openai_client(api_key)

    file_bytes = uploaded_pdf.read()
    pages = page_images_from_pdf(file_bytes, dpi=220)

    prog = st.progress(0.0, text="Starting…")

    all_detections: List[Dict] = []
    total = len(pages)

    for i, page_img in enumerate(pages, start=1):
        prog.progress(i / total, text=f"Detecting welds on page {i}/{total}…")
        detections = gpt_detect_welds(client, page_img, mark_ambiguous=mark_ambiguous)

        # Optional: show crop previews (simple dots here, could draw boxes if you add sizes)
        if show_crops and detections:
            st.markdown(f"**Page {i} previews**")
            # Just render the full image for now; advanced: crop thumbnails around (x,y)
            st.image(page_img, caption=f"Page {i} (raw)")

        for d in detections:
            d["page"] = i
        all_detections.extend(detections)

    prog.empty()

    if not all_detections:
        st.warning("No weld tags detected (or parsing failed). Try another PDF or increase DPI.")
    else:
        st.success(f"Detected {sum(1 for d in all_detections if d.get('weld_number') is not None)} weld numbers "
                   f"(+ {sum(1 for d in all_detections if d.get('weld_number') is None)} ambiguous).")

        import pandas as pd
        df = pd.DataFrame(all_detections, columns=["page", "weld_number", "x", "y", "ambiguous", "note"])
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download detections (CSV)", csv, file_name="weld_detections.csv", mime="text/csv")
