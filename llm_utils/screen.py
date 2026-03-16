import pyautogui
import base64
from io import BytesIO
from langchain.tools import tool
from .config_loader import wrap_output

@tool
def take_screenshot(save_path: str = "screenshot.png"):
    """Takes a screenshot of the current screen and saves it to the specified path."""
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save(save_path)
        return wrap_output(f"Screenshot saved to {save_path}")
    except Exception as e:
        return wrap_output(f"Failed to take screenshot: {str(e)}")



def get_screenshot_base64() -> str:
    """Takes a screenshot and returns it as a base64 encoded string."""
    buffered = BytesIO()
    screenshot = pyautogui.screenshot()
    screenshot.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_screen_size() -> tuple[int, int]:
    """Returns the screen width and height."""
    return pyautogui.size()
