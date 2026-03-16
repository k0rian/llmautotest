import pyautogui
import time
from dataclasses import dataclass
from .config_loader import wrap_output
from langchain.tools import tool

# Set a pause to give some safety margin (though langchain tools might not need it as much as direct scripts)
pyautogui.PAUSE = 0.5

@tool
def click(x: int, y: int):
    """点击屏幕上的指定坐标 (Left Click)"""
    try:
        pyautogui.click(x, y)
        return wrap_output(f"Successfully clicked at ({x}, {y})")
    except Exception as e:
        return wrap_output(f"Failed to click at ({x}, {y}): {str(e)}")

@tool
def double_click(x: int, y: int):
    """双击屏幕上的指定坐标 (Double Click)"""
    try: 
        pyautogui.doubleClick(x, y)
        return wrap_output(f"Successfully double-clicked at ({x}, {y})")
    except Exception as e:
        return wrap_output(f"Failed to double-click at ({x}, {y}): {str(e)}")

@tool
def right_click(x: int, y: int):
    """右击屏幕上的指定坐标 (Right Click)"""
    try:
        pyautogui.rightClick(x, y)
        return wrap_output(f"Successfully right-clicked at ({x}, {y})")
    except Exception as e:
        return wrap_output(f"Failed to right-click at ({x}, {y}): {str(e)}")

@tool
def drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0):
    """从起始坐标拖拽到终点坐标"""
    try:
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration, button='left')
        return wrap_output(f"Successfully dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
    except Exception as e:
        return wrap_output(f"Failed to drag: {str(e)}")

@tool
def type_text(text: str, enter: bool = False):
    """在当前焦点位置输入文本"""
    try:
        # typewrite handles individual characters better for some apps
        pyautogui.write(text, interval=0.05) 
        if enter:
            pyautogui.press('enter')
        return wrap_output(f"Successfully typed: {text}" + (" [Enter]" if enter else ""))
    except Exception as e:
        return wrap_output(f"Failed to type text: {str(e)}")

@tool
def hotkey(keys: str):
    """
    执行热键组合。
    keys 格式例如: 'ctrl+c', 'alt+tab', 'ctrl+shift+esc'
    """
    try:
        key_list = keys.split('+')
        key_list = [k.strip().lower() for k in key_list]
        pyautogui.hotkey(*key_list)
        return wrap_output(f"Successfully pressed hotkey: {keys}")
    except Exception as e:
        return wrap_output(f"Failed to press hotkey {keys}: {str(e)}")

@tool
def scroll(amount: int):
    """
    滚动屏幕。
    amount > 0: 向上滚动
    amount < 0: 向下滚动
    """
    try:
        pyautogui.scroll(amount)
        return wrap_output(f"Successfully scrolled {amount}")
    except Exception as e:
        return wrap_output(f"Failed to scroll: {str(e)}")

@tool
def get_cursor_position():
    """获取当前鼠标位置"""
    try:
        x, y = pyautogui.position()
        return wrap_output(f"Current cursor position: ({x}, {y})")
    except Exception as e:
        return wrap_output(f"Failed to get cursor position: {str(e)}")

