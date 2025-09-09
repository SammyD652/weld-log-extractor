"""
Optional module to validate or enrich ambiguous weld rows using a large
language model (LLM), such as OpenAI's Vision models. This is a stub; by
default it returns the input records unchanged. When you observe recurring
mistakes or ambiguous detections, implement calls to the OpenAI API here.

Guidance:
 - Only call the LLM when necessary to keep usage low. You might batch
   ambiguous rows and provide relevant page snippets for context.
 - Use st.secrets to access the API key (set in Streamlit Cloud settings).
"""

from typing import List

import streamlit as st

from extractor.schema import WeldRecord


def validate_with_llm_if_enabled(
    records: List[WeldRecord], pages: List[dict]
) -> List[WeldRecord]:
    """
    Placeholder validation function.

    Simply returns the records unchanged. Extend this function to call a
    large language model to verify or correct extracted fields when
    heuristics are insufficient.
    """
    # Example skeleton for future implementation:
    # from openai import OpenAI
    # client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    # Build a prompt with ambiguous records and page snippets...
    # response = client.chat.completions.create(...)
    # Use the response to adjust fields.
    return records
