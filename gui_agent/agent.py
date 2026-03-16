from numpy.lib._npyio_impl import _save_dispatcher
from ui_tars.action_parser import parsing_response_to_pyautogui_code
import pyautogui
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",  # vLLM 启动的 OpenAI 兼容接口
)

def get_screen_size() -> tuple[int, int]:
    """Returns the screen width and height."""
    return pyautogui.size()

class GUIAgent:
    def __init__(self):
        self.screen_width, self.screen_height = get_screen_size()
    
    def __take_screenshot__(self,save_path='/tmp/screenshot/'):
        """Captures a screenshot and returns it as a base64-encoded string."""
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(save_path)
            return save_path
        except Exception as e:
            return f"Failed to take screenshot: {str(e)}"

    def __encode_image__(self,img_path: str):
        """Encodes an image to base64."""
        try:
            with open(img_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            return f"Failed to encode image: {str(e)}"

    def __get_vlm_action__(self,instruction: str):
        prompt = '''
    You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

    ## Output Format
    ```
    Thought: ...
    Action: ...
    ```

    ## Action Space

    click(point='<point>x1 y1</point>')
    left_double(point='<point>x1 y1</point>')
    right_single(point='<point>x1 y1</point>')
    drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
    hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
    type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
    scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
    wait() #Sleep for 5s and take a screenshot to check for any changes.
    finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


    ## Note
    - Use Chinese in `Thought` part.
    - Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

    ## User Instruction
    {instruction}
        '''.format(instruction=instruction)
        screenshot_path = self.__take_screenshot__()
        encoded_image = self.__encode_image__(screenshot_path)
        response = client.chat.completions.create(
            model="ui-tars",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0,
        )
        action = response.choices[0].message.content
        return action

    def _execute_action(self, action_code):
        """执行 pyautogui 自动化代码"""
        try:
            # 执行生成的操作代码
            exec(action_code)
            return {"status": "success", "message": "操作执行成功"}
        except Exception as e:
            return {"status": "failed", "message": f"操作执行失败：{str(e)}"}

    def run(self, main_instruction, max_steps=10):
        """
        核心运行逻辑：拆解主指令为分步操作，循环执行直到完成/超时
        :param main_instruction: 主 Agent 下发的自然语言指令
        :param max_steps: 最大执行步数（防止无限循环）
        :return: 最终执行结果
        """
        step = 0
        while step < max_steps:
            step += 1
            print(f"\n=== 执行第 {step} 步 ===")
            
            # 1. 截取当前屏幕
            screenshot_path = self.__take_screenshot__()
            
            # 2. 调用 VLM 生成单步操作指令
            vlm_response = self.__get_vlm_action__(main_instruction, screenshot_path)
            print(f"VLM 输出：\n{vlm_response}")
            
            # 3. 解析 VLM 输出为结构化数据
            try:
                parsed_dict = parse_action_to_structure_output(
                    vlm_response,
                    factor=1000,  # UI-TARS 模型默认的坐标基准
                    origin_resized_height=self.screen_height,
                    origin_resized_width=self.screen_width,
                    model_type="qwen25vl"  # 适配 Qwen2-VL 模型输出格式
                )
            except Exception as e:
                print(f"解析操作指令失败：{e}")
                continue
            
            # 4. 生成 pyautogui 可执行代码
            if not parsed_dict:
                print("未解析到有效操作，终止")
                break
            action_code = parsing_response_to_pyautogui_code(
                parsed_dict,
                image_height=self.screen_height,
                image_width=self.screen_width
            )
            print(f"生成的自动化代码：\n{action_code}")
            
            # 5. 执行操作
            exec_result = self._execute_action(action_code)
            print(f"执行结果：{exec_result}")
            
            # 6. 检查是否任务完成
            if "finished" in vlm_response:
                finish_content = vlm_response.split("finished(content='")[1].split("')")[0]
                return {
                    "status": "completed",
                    "step": step,
                    "message": finish_content
                }
        
        return {
            "status": "timeout",
            "step": step,
            "message": f"达到最大步数 {max_steps}，任务未完成"
        }