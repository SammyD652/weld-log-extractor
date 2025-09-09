# Weld Log Extraction (Online Only)

Fully online Streamlit app that reads **isometric PDF drawings** and produces a weld log (**Excel/CSV**) with:

- **Weld Number**
- **Joint Size** (ND/DN)
- **Joint Type** (BW/SW/THD/SO/etc.)
- **Material Description**

## How it works
1. Extract text per page via **pdfminer.six** (fast).
2. Pages with little/no text are rendered to images and OCR’d with **Tesseract** (via **pdf2image**).
3. Heuristics parse weld numbers, ND/DN sizes, joint type codes, and material text (from BOM/spec sections).
4. Pydantic schema validation; optional **OpenAI Vision** validation (toggle).

## Deploy on Streamlit Cloud

### One-time GitHub setup
1. Create a new repo (or reuse) named `weld-log-extractor` on GitHub.
2. Add all files from this project (`app.py`, `extractor/`, `utils/`, `requirements.txt`, `packages.txt`, `README.md`, `.streamlit/secrets.toml`).
3. Commit to `main`.

### Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud) → **New app**.
2. Connect your GitHub, choose the repo and branch (`main`), set **Main file path** to `app.py`.
3. Click **Deploy**.

### Secrets (optional for LLM validation)
In Streamlit Cloud:

1. App → **⚙️ Settings** → **Secrets**.
2. Add:
   ``
   OPENAI_API_KEY = "sk-..."
   ``
3. Click **Save** and **Rerun** the app.

## Usage

1. Upload one or more **PDF** isometrics.
2. Inspect the preview table.
3. Download the weld log as **Excel** or **CSV**.

## Notes

- On Streamlit Cloud, system packages `tesseract-ocr` and `poppler-utils` are installed via `packages.txt`.
- If a page has a proper text layer, OCR is skipped for speed.
- Parsing is heuristic and can be refined for your drawing standards.
