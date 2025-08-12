import os
import io
import base64
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import streamlit as st

# Streamlit application to extract a weld log from isometric drawing PDFs using
# OpenAI's GPT‑4o Vision model. This version attempts to read an API key from
# Streamlit's secrets configuration (``.streamlit/secrets.toml``) or from the
# ``OPENAI_API_KEY`` environment variable. If neither are set, the user may
# provide the key via the sidebar input. The resulting weld log is written to
# an Excel file and made available for download.
#
# To run on Streamlit Community Cloud without pasting the key each time,
# create a file at ``.streamlit/secrets.toml`` in your repository with the
# following content:
#
#     OPENAI_API_KEY = "sk-..."
#
# The key will be injected into ``st.secrets`` at runtime.


# ============ UI ============

# Configure the page layout and title
st.set_page_config(page_title="Weld Log Extractor (GPT‑4o Vision)", layout="wide")
st.title("Weld Log Extractor — GPT‑4o Vision")

# Fetch an API key from secrets or environment as a default
api_key_default: str | None = None
try:
    # st.secrets is only defined on Streamlit Cloud when ``secrets.toml`` exists
    api_key_default = st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
except Exception:
    api_key_default = None
if not api_key_default:
    api_key_default = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.header("1) API Key")
    # Inform user if a key has been pre‑configured
    if api_key_default:
        st.write("Using API key from Streamlit secrets or environment.")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=api_key_default or "",
        placeholder="sk-...",
        help="Get this from platform.openai.com → View API keys",
    )

    st.header("2) Model & Pages")
    model = st.selectbox("Vision model", ["gpt-4o"], index=0)
    max_pages = st.number_input("Max PDF pages to read", min_value=1, max_value=15, value=3, step=1)
    dpi = st.slider("Image render DPI (higher = sharper, slower)", 100, 300, 220)

    st.header("3) Output")
    excel_filename = st.text_input("Excel file name", value="weld_log.xlsx")

# Prompt user to upload a PDF
st.markdown("**Upload your isometric drawing PDF (full drawing, no snips).**")
uploaded = st.file_uploader("PDF file", type=["pdf"])

# ============ Helpers ============

def pdf_pages_to_images(pdf_bytes: bytes, max_pages: int, dpi: int) -> list[Image.Image]:
    """
    Convert the first N pages of a PDF into a list of PIL images at the given DPI.
    A zoom factor converts DPI (dots per inch) to the internal 72 DPI of PDF
    rendering. Alpha channels are discarded to simplify downstream processing.

    :param pdf_bytes: Raw PDF data.
    :param max_pages: Maximum number of pages to render.
    :param dpi: Target dots per inch for rendering.
    :returns: List of PIL Image objects.
    """
    images: list[Image.Image] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = min(len(doc), max_pages)
    for pno in range(page_count):
        page = doc.load_page(pno)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    return images

def pil_to_data_url(img: Image.Image) -> str:
    """
    Convert a PIL Image into a base64‑encoded data URL that OpenAI's
    Vision API accepts.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_extraction_prompt() -> str:
    """
    Construct the system prompt for the GPT model to extract weld log data.
    The prompt defines a strict JSON schema and guidelines for interpreting
    weld tags, joint sizes, joint types, and material descriptions. The model
    must return compact JSON without commentary.
    """
    return (
        "You are a welding QA assistant. Read the isometric drawing images and extract a weld log.\n"
        "Return strict JSON only (no commentary). JSON schema:\n"
        "{\n"
        '  "welds": [\n'
        "    {\n"
        '      "weld_number": "string",\n'
        '      "joint_size": "string",\n'
        '      "joint_type": "string",\n'
        '      "material_description": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Notes:\n"
        "- Weld numbers are on tags near joints (e.g., SW-### or FW-###). If tag shows just a number, return that.\n"
        "- Joint size comes from ND/size callouts or weld symbol (e.g., ND25, 2\"), keep units as seen.\n"
        "- Joint type from the symbol/legend (e.g., butt, fillet, socket weld). If unclear, infer from symbols and note best guess.\n"
        "- Material description comes from the BOM/spec table (e.g., carbon steel, SS316L, schedule, rating). If multiple, pick most specific.\n"
        "- If an item is missing, put an empty string rather than inventing data.\n"
        "- Return compact JSON. Do NOT include markdown or explanations."
    )

def parse_json_loose(text: str) -> dict:
    """
    Extract a JSON object from text even if the model wraps it with extra
    commentary. It finds the first brace pair and attempts to load it with
    ``json.loads``, returning an empty weld list on failure.

    :param text: Raw string from the model.
    :returns: Dictionary containing a ``welds`` key.
    """
    import json
    import re
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"welds": []}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"welds": []}

# ============ Run ============

run_btn = st.button(
    "Extract Weld Log",
    type="primary",
    disabled=not (uploaded and api_key),
)

if uploaded:
    st.write(f"**Selected file:** {uploaded.name}")
    pdf_bytes = uploaded.read()
    with st.expander("Preview (first few pages)"):
        preview_cols = st.columns(2)
        images = pdf_pages_to_images(pdf_bytes, max_pages=int(max_pages), dpi=int(dpi))
        # Display preview images across two columns.
        # `st.image` accepts ``use_column_width`` to scale the image to the column width.  
        # Using ``use_container_width`` here will raise a TypeError.
        for i, img in enumerate(images):
            # Each image is displayed within its own column context.
            with preview_cols[i % 2]:
                st.image(img, caption=f"Page {i+1}", use_column_width=True)

if run_btn:
    if not api_key:
        st.error("Please provide an OpenAI API key in the sidebar or via secrets.")
        st.stop()
    if not uploaded:
        st.error("Please upload a PDF.")
        st.stop()

    st.info("Rendering PDF pages and sending to GPT‑4o Vision… This can take ~10–30 seconds depending on DPI & pages.")

    images = pdf_pages_to_images(pdf_bytes, max_pages=int(max_pages), dpi=int(dpi))
    if not images:
        st.error("No pages found to process.")
        st.stop()

    # Build messages for Responses API style payload
    prompt = build_extraction_prompt()
    content: list[dict] = [
        {"type": "text", "text": prompt},
    ]
    for img in images:
        content.append({
            "type": "input_image",
            "image_url": pil_to_data_url(img),
        })

    # === Call OpenAI ===
    os.environ["OPENAI_API_KEY"] = api_key
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Optional: JSON schema enforcement to keep output clean
    json_schema = {
        "name": "weld_log_schema",
        "schema": {
            "type": "object",
            "properties": {
                "welds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "weld_number": {"type": "string"},
                            "joint_size": {"type": "string"},
                            "joint_type": {"type": "string"},
                            "material_description": {"type": "string"},
                        },
                        "required": [
                            "weld_number",
                            "joint_size",
                            "joint_type",
                            "material_description",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["welds"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
        raw = resp.output_text if hasattr(resp, "output_text") else str(resp)
    except Exception as e:
        st.exception(e)
        st.stop()

    # Parse JSON (be tolerant)
    data = parse_json_loose(raw)
    welds = data.get("welds", [])

    if not welds:
        st.warning("No welds were extracted. Try increasing DPI or pages, or upload a clearer PDF.")
        st.text_area("Model raw output (debug)", raw, height=200)
        st.stop()

    df = pd.DataFrame(welds)
    st.success(f"Found {len(df)} weld(s).")
    st.dataframe(df, use_container_width=True)

    # Download Excel
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Weld Log")
        xlsx_bytes = buffer.getvalue()

    st.download_button(
        label=f"Download {excel_filename}",
        data=xlsx_bytes,
        file_name=excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("See JSON"):
        st.code(data, language="json")
