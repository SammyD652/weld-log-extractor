"""
Utility functions for exporting DataFrames to Excel and CSV.

Using pandas to write Excel provides broad compatibility (xlsxwriter engine).
We calculate column widths based on content length for better formatting in
exported spreadsheets.
"""

import io
import pandas as pd


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Return a bytes object containing the Excel representation of the DataFrame."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            width = max(12, min(40, int(df[col].astype(str).str.len().max() or 12) + 2))
            worksheet.set_column(idx, idx, width)
    return output.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return a bytes object containing the CSV representation of the DataFrame."""
    return df.to_csv(index=False).encode("utf-8")
