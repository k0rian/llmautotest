import json
import re
from openai import OpenAI
from .screen import get_screenshot_base64, get_screen_size
from langchain.tools import tool
from .config_loader import get_client, wrap_output
from .ocr import ocr_candidates_from_base64
from pydantic import BaseModel, Field

client = get_client()

class ElementLocation(BaseModel):
    bbox: list[int] = Field(description="The bounding box of the element [xmin, ymin, xmax, ymax] normalized to 0-1000 space")

# @tool
# def get_element_coordinates(instruction: str):
#     """
#     Analyzes the current screen to find the element described by the instruction.
#     Returns a dictionary with 'x' and 'y' coordinates.
#     """
#     screenshot_base64 = get_screenshot_base64()
#     width, height = get_screen_size()
    
#     prompt = f"""
#     You are a GUI automation assistant.
#     Look at the screenshot and identify the UI element described by the user.
#     Return the bounding box of the element in [xmin, ymin, xmax, ymax] format, normalized to a 0-1000 coordinate space.
    
#     User instruction: "{instruction}"
#     """
    
#     try:
#         response = client.beta.chat.completions.parse(
#             model="doubao-seed-1-6-flash-250828",
#             # stream=True, # Changed to False for simplicity in tool call
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/png;base64,{screenshot_base64}"
#                             },
#                         },
#                     ],
#                 }
#             ],
#             response_format=ElementLocation,
#             extra_body={
#                 "thinking": {
#                     "type": "disabled"
#                 }
#             }
#         )
#         resp = response.choices[0].message.parsed
#         bbox = resp.bbox
        
#         if not bbox or len(bbox) != 4:
#             return {"error": "Invalid bbox returned"}
            
#         # Convert normalized [0-1000] bbox to screen coordinates
#         xmin, ymin, xmax, ymax = bbox
        
#         # Calculate center
#         center_x = (xmin + xmax) / 2 / 1000 * width
#         center_y = (ymin + ymax) / 2 / 1000 * height
        
#         print(f"Vision result: bbox={bbox}, screen_size={width}x{height}, target=({center_x}, {center_y})")
        
#         return wrap_output({"x": int(center_x), "y": int(center_y)})
#     except Exception as e:
#         print(f"Vision error: {e}")
#         return wrap_output({"error": str(e)})
@tool
def get_element_coordinates(instruction: str):
    """
    Use OCR + Vision model to locate UI element more accurately.
    """
    screenshot_base64 = get_screenshot_base64()
    width, height = get_screen_size()

    # 1️⃣ OCR 本地感知
    ocr_candidates = ocr_candidates_from_base64(
        screenshot_base64,
        width,
        height
    )

    # 只保留前 N 个，避免 prompt 爆炸
    ocr_candidates = ocr_candidates[:30]

    # 2️⃣ 构造增强 Prompt
    prompt = f"""
You are a GUI automation assistant.

User wants to interact with the following UI element:
"{instruction}"

We already ran OCR on the screen and detected the following candidate elements
(each bbox is normalized to 0-1000 space):

{json.dumps(ocr_candidates, ensure_ascii=False, indent=2)}

Instructions:
- Prefer selecting one of the OCR candidates if it matches the user instruction.
- If none match well, infer the element visually from the screenshot.
- Return the FINAL bounding box [xmin, ymin, xmax, ymax] normalized to 0-1000.
- Do NOT invent coordinates unrelated to the screen.
"""

    try:
        response = client.beta.chat.completions.parse(
            model="doubao-seed-1-6-flash-250828",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            },
                        },
                    ],
                }
            ],
            response_format=ElementLocation,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

        resp = response.choices[0].message.parsed
        bbox = resp.bbox

        if not bbox or len(bbox) != 4:
            return wrap_output({"error": "Invalid bbox returned"})

        xmin, ymin, xmax, ymax = bbox

        center_x = int((xmin + xmax) / 2 / 1000 * width)
        center_y = int((ymin + ymax) / 2 / 1000 * height)

        print(
            f"[VISION+OCR] bbox={bbox}, "
            f"screen={width}x{height}, "
            f"center=({center_x},{center_y})"
        )

        return wrap_output({
            "x": center_x,
            "y": center_y
        })

    except Exception as e:
        print(f"Vision error: {e}")
        return wrap_output({"error": str(e)})
