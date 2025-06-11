# This script is intended for use in a Streamlit environment.
# To run it locally: pip install streamlit openai pillow pymupdf pandas

try:
    import streamlit as st
    import openai
    from PIL import Image
    import pandas as pd
    import io
    import fitz  # PyMuPDF for PDF to image
except ModuleNotFoundError as e:
    print(f"❌ Missing module: {e.name}. Please install dependencies before running.")
else:
    st.set_page_config(page_title="GPT-4o Weld Log Extractor")

    st.title("🧠 GPT-4o Weld Log Extractor")

    uploaded_file = st.file_uploader("Upload a drawing (PDF or image)", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        file_bytes = uploaded_file.read()

        # Convert PDF to image
        if uploaded_file.type == "application/pdf":
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            except Exception as e:
                st.error(f"Failed to read PDF: {e}")
                st.stop()
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                st.stop()

        st.image(img, caption="Uploaded Drawing", use_column_width=True)

        if st.button("Extract Weld Log"):
            st.info("Sending image to GPT-4o...")

            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]

                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a welding QA/QC assistant. Extract weld numbers, joint sizes, joint types, and material descriptions from engineering drawings. Respond in CSV format."},
                        {"role": "user", "content": "Extract a weld log from this image."}
                    ],
                    images=[img]
                )

                extracted_text = response.choices[0].message.content
                st.text_area("Extracted Raw Output", extracted_text, height=300)

                # Attempt to parse CSV
                try:
                    rows = [row.split(",") for row in extracted_text.strip().split("\n") if "," in row]
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                    st.dataframe(df)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download as CSV", csv, "weld_log.csv", "text/csv")
                except Exception as parse_error:
                    st.warning("Could not parse structured table from response.")
                    st.exception(parse_error)

            except Exception as e:
                st.error(f"OpenAI call failed: {e}")
