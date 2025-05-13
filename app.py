import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import pandas as pd
import math
import io

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def pdf_page_to_image(pdf_bytes, page_number=0, zoom=2):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def auto_assign_welds_to_bom(image, df_weld_types, df_bom, max_distance_threshold=150):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    weld_tags = data[data['text'].str.match(r'^\d{2,4}$', na=False)]
    bom_tags = data[data['text'].str.match(r'^\d{1,2}$', na=False)]

    assignments = []
    for _, weld in weld_tags.iterrows():
        weld_pos = (weld['left'] + weld['width'] // 2, weld['top'] + weld['height'] // 2)
        weld_num = weld['text']
        closest_bom = None
        min_dist = float('inf')
        for _, bom in bom_tags.iterrows():
            bom_pos = (bom['left'] + bom['width'] // 2, bom['top'] + bom['height'] // 2)
            dist = distance(weld_pos, bom_pos)
            if dist < min_dist:
                min_dist = dist
                closest_bom = bom['text']
        if min_dist <= max_distance_threshold:
            assignments.append({"Weld Number": weld_num, "BOM ID": closest_bom})
        else:
            assignments.append({"Weld Number": weld_num, "BOM ID": None})

    df_assignments = pd.DataFrame(assignments)
    df_welds = pd.merge(df_weld_types, df_assignments, on="Weld Number", how="inner")
    df_final = pd.merge(df_welds, df_bom, left_on="BOM ID", right_on="ID", how="left")
    df_final_weld_log = df_final.loc[:, ["Weld Number", "Weld Type", "ND", "Description"]].copy()
    df_final_weld_log.rename(columns={
        "ND": "Joint Size",
        "Description": "Material Description"
    }, inplace=True)
    return df_final_weld_log

def main():
    st.title("Piping Isometric Weld Log Extractor (PyMuPDF Version)")

    st.markdown("Upload your PDF piping drawing and get back an Excel weld log.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        zoom = st.slider("Zoom factor for image extraction", 1, 4, 2)
        max_dist = st.slider("Max Distance Threshold (px)", 50, 300, 150)

        image = pdf_page_to_image(uploaded_file.read(), page_number=0, zoom=zoom)

        st.image(image, caption="Drawing Preview", use_column_width=True)

        st.subheader("Enter Weld Types")
        weld_numbers = st.text_input("Weld Numbers (comma-separated)", "001,002,003,004")
        weld_type_default = st.selectbox("Default Weld Type", ["Shop", "Field"])

        weld_list = [w.strip() for w in weld_numbers.split(",") if w.strip()]
        df_weld_types = pd.DataFrame({
            "Weld Number": weld_list,
            "Weld Type": [weld_type_default] * len(weld_list)
        })

        st.subheader("Upload BOM Table CSV")
        bom_file = st.file_uploader("Upload BOM CSV", type="csv")

        if bom_file:
            df_bom = pd.read_csv(bom_file)
            st.write("BOM Table:", df_bom)

            if st.button("Run Extraction"):
                result = auto_assign_welds_to_bom(image, df_weld_types, df_bom, max_distance_threshold=max_dist)
                st.write("Weld Log:", result)

                # Download Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result.to_excel(writer, index=False)
                st.download_button(
                    label="Download Weld Log as Excel",
                    data=output.getvalue(),
                    file_name="weld_log_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()

