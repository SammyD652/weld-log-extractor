import base64
from openai import OpenAI

def image_to_base64(uploaded_image):
    """
    Convert uploaded image file to base64 string.
    """
    return base64.b64encode(uploaded_image.read()).decode("utf-8")

def get_gpt_vision_comparison(api_key, submittal_text, image_base64):
    """
    Use OpenAI GPT-4o Vision to compare technical submittal text with equipment nameplate image.
    Returns a markdown-formatted table of compliance results.
    """
    client = OpenAI(api_key=api_key)

    system_msg = {
        "role": "system",
        "content": "You are a mechanical QAQC engineer. Compare the equipment nameplate image to the technical submittal and identify any mismatches. Respond with a markdown table: Field | Submittal Value | Equipment Value | Match (✅/❌)"
    }

    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Technical Submittal:\n\n{submittal_text}"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{image_base64}",
                "detail": "high"
            }}
        ]
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_msg, user_msg],
        temperature=0.2
    )

    return response.choices[0].message.content
