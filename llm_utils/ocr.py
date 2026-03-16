import pytesseract
import cv2
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import cv2

def detect_buttons_cv(
    screenshot_base64: str,
    screen_w: int,
    screen_h: int
):
    """
    Detect clickable button-like regions using OpenCV.
    Returns normalized bbox [xmin, ymin, xmax, ymax] in 0-1000 space.
    """
    img_bytes = base64.b64decode(screenshot_base64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 几何过滤
        if w < 80 or h < 30 or w > 500 or h > 200:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < 1.2 or aspect_ratio > 8:
            continue

        area = cv2.contourArea(cnt)
        if area < 1500:
            continue

        # 归一化到 0-1000
        xmin = int(x / screen_w * 1000)
        ymin = int(y / screen_h * 1000)
        xmax = int((x + w) / screen_w * 1000)
        ymax = int((y + h) / screen_h * 1000)

        candidates.append({
            "type": "button",
            "source": "cv",
            "bbox": [xmin, ymin, xmax, ymax],
            "center": [
                int((xmin + xmax) / 2),
                int((ymin + ymax) / 2)
            ],
            "confidence_hint": 0.5
        })

    return candidates

def ocr_candidates_from_base64(screenshot_base64: str, screen_w: int, screen_h: int):
    """
    Run OCR on screenshot and return candidates with normalized bbox (0-1000).
    """
    img_bytes = base64.b64decode(screenshot_base64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    data = pytesseract.image_to_data(
        img_cv,
        output_type=pytesseract.Output.DICT,
        lang="chi_sim+eng"
    )

    candidates = []

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        # 过滤明显不是 UI 的噪声
        if w < 20 or h < 15:
            continue

        xmin = int(x / screen_w * 1000)
        ymin = int(y / screen_h * 1000)
        xmax = int((x + w) / screen_w * 1000)
        ymax = int((y + h) / screen_h * 1000)

        candidates.append({
            "text": text,
            "bbox": [xmin, ymin, xmax, ymax],
            "center": [
                int((xmin + xmax) / 2),
                int((ymin + ymax) / 2)
            ]
        })

    return candidates