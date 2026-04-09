import os
import re
import time
import base64
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, Literal, Any

import pyautogui
from PIL import Image

from openai import OpenAI
from vllm import LLM, SamplingParams

from ui_tars.action_parser import parsing_response_to_pyautogui_code ,parse_action_to_structure_output

def get_screen_size() -> tuple[int, int]:
    return pyautogui.size()


@dataclass
class AgentStep:
    step: int
    instruction: str
    vlm_response: str
    parsed_action: Optional[dict] = None
    action_code: Optional[str] = None
    exec_result: Optional[dict] = None


@dataclass
class GUIAgentConfig:
    model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B"
    backend: Literal["local", "server"] = "local"

    # server 模式参数
    server_base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"

    # vLLM 本地模式参数
    trust_remote_code: bool = True
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096

    # 生成参数
    temperature: float = 0.0
    max_tokens: int = 512

    # Agent 行为参数
    max_history: int = 8
    screenshot_dir: Optional[str] = None   # 为 None 时仅内存处理；给目录则可选保存调试截图
    save_debug_screenshot: bool = False


class GUIAgent:
    def __init__(self, config: Optional[GUIAgentConfig] = None):
        self.config = config or GUIAgentConfig()
        self.screen_width, self.screen_height = get_screen_size()

        self.client: Optional[OpenAI] = None
        self.llm: Optional[LLM] = None

        self.history: list[AgentStep] = []

        self._init_backend()

    # =========================
    # 初始化
    # =========================
    def _init_backend(self) -> None:
        if self.config.backend == "local":
            self.llm = LLM(
                model=self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                dtype=self.config.dtype,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
            )
        elif self.config.backend == "server":
            self.client = OpenAI(
                base_url=self.config.server_base_url,
                api_key=self.config.api_key,
            )
            self._check_server_ready()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _check_server_ready(self) -> None:
        try:
            assert self.client is not None
            self.client.models.list()
        except Exception as e:
            raise RuntimeError(
                f"无法连接到 vLLM 服务: {self.config.server_base_url}, error={e}"
            )

    # =========================
    # 截图
    # =========================
    def _take_screenshot(self) -> Image.Image:
        img = pyautogui.screenshot()

        if self.config.save_debug_screenshot and self.config.screenshot_dir:
            os.makedirs(self.config.screenshot_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            save_path = os.path.join(self.config.screenshot_dir, f"screenshot_{ts}.png")
            img.save(save_path)

        return img

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_prompt(self, instruction: str) -> str:
        """
        这里你替换成自己的 Prompt 组织逻辑。
        下面只是一个最小占位实现。
        """
        history_text = self._format_history_for_prompt()
        prompt = f"""
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
        """.strip()
        return prompt

    def _format_history_for_prompt(self) -> str:
        if not self.history:
            return "无"

        recent_history = self.history[-self.config.max_history:]
        lines = []
        for item in recent_history:
            lines.append(
                f"Step {item.step}: "
                f"vlm_response={item.vlm_response!r}, "
                f"exec_result={item.exec_result}"
            )
        return "\n".join(lines)

    # =========================
    # 模型推理
    # =========================
    def _get_vlm_action(self, instruction: str) -> str:
        screenshot = self._take_screenshot()
        prompt = self._build_prompt(instruction)

        if self.config.backend == "local":
            return self._get_vlm_action_local(prompt, screenshot)
        return self._get_vlm_action_server(prompt, screenshot)

    def _get_vlm_action_local(self, prompt: str, screenshot: Image.Image) -> str:
        """
        本地 vLLM LLM 模式
        注意：
        1. 多模态 prompt 模板必须和具体模型匹配
        2. 如果 UI-TARS 仓库定义了专门的 chat template，优先按模型仓库示例改这里
        """
        assert self.llm is not None

        inputs = [{
            "prompt": f"USER: <image>\n{prompt}\nASSISTANT:",
            "multi_modal_data": {
                "image": screenshot
            }
        }]

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _get_vlm_action_server(self, prompt: str, screenshot: Image.Image) -> str:
        """
        走 vllm serve 提供的 OpenAI-compatible API
        """
        assert self.client is not None

        encoded_image = self._encode_image_to_base64(screenshot)

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()

    # =========================
    # Action 解析
    # =========================
    def _parse_action(self, vlm_response: str) -> dict:
        """
        你需要替换成你项目里的真实解析函数。
        """
        parsed_dict = parse_action_to_structure_output(
            vlm_response,
            factor=1000,
            origin_resized_height=self.screen_height,
            origin_resized_width=self.screen_width,
            model_type="qwen25vl",
        )
        return parsed_dict

    def _extract_finished_content(self, text: str) -> Optional[str]:
        patterns = [
            r"finished\(content='(.*?)'\)",
            r'finished\(content="(.*?)"\)',
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1)
        return None

    # =========================
    # 执行
    # =========================
    def _execute_action(self, action_code: str) -> dict:
        """
        仍然使用 exec，但至少限制作用域。
        如果你后面愿意，我更建议把 action_code 改成结构化 action 再分发执行。
        """
        safe_globals = {
            "__builtins__": {
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "abs": abs,
                "print": print,
            },
            "pyautogui": pyautogui,
            "time": time,
        }

        safe_locals = {}

        try:
            exec(action_code, safe_globals, safe_locals)
            return {"status": "success", "message": "操作执行成功"}
        except Exception as e:
            return {"status": "failed", "message": f"操作执行失败: {e}"}

    # =========================
    # 单步执行
    # =========================
    def step(self, instruction: str, step_num: int) -> dict:
        vlm_response = self._get_vlm_action(instruction)
        print(f"\n=== 执行第 {step_num} 步 ===")
        print(f"VLM 输出：\n{vlm_response}")

        finished_content = self._extract_finished_content(vlm_response)
        if finished_content is not None:
            step_obj = AgentStep(
                step=step_num,
                instruction=instruction,
                vlm_response=vlm_response,
                exec_result={"status": "completed", "message": finished_content},
            )
            self.history.append(step_obj)
            return {
                "status": "completed",
                "step": step_num,
                "message": finished_content,
                "vlm_response": vlm_response,
            }

        try:
            parsed_dict = self._parse_action(vlm_response)
        except Exception as e:
            step_obj = AgentStep(
                step=step_num,
                instruction=instruction,
                vlm_response=vlm_response,
                exec_result={"status": "failed", "message": f"解析失败: {e}"},
            )
            self.history.append(step_obj)
            return {
                "status": "failed",
                "step": step_num,
                "message": f"解析操作指令失败: {e}",
                "vlm_response": vlm_response,
            }

        if not parsed_dict:
            step_obj = AgentStep(
                step=step_num,
                instruction=instruction,
                vlm_response=vlm_response,
                parsed_action=parsed_dict,
                exec_result={"status": "failed", "message": "未解析到有效操作"},
            )
            self.history.append(step_obj)
            return {
                "status": "failed",
                "step": step_num,
                "message": "未解析到有效操作",
                "vlm_response": vlm_response,
            }

        action_code = parsing_response_to_pyautogui_code(
            parsed_dict,
            image_height=self.screen_height,
            image_width=self.screen_width,
        )
        print(f"生成的自动化代码：\n{action_code}")

        exec_result = self._execute_action(action_code)
        print(f"执行结果：{exec_result}")

        step_obj = AgentStep(
            step=step_num,
            instruction=instruction,
            vlm_response=vlm_response,
            parsed_action=parsed_dict,
            action_code=action_code,
            exec_result=exec_result,
        )
        self.history.append(step_obj)

        return {
            "status": exec_result["status"],
            "step": step_num,
            "message": exec_result["message"],
            "vlm_response": vlm_response,
            "parsed_action": parsed_dict,
            "action_code": action_code,
        }

    # =========================
    # 主循环
    # =========================
    def run(self, main_instruction: str, max_steps: int = 10) -> dict:
        for step_num in range(1, max_steps + 1):
            result = self.step(main_instruction, step_num)

            if result["status"] == "completed":
                return result

        return {
            "status": "timeout",
            "step": max_steps,
            "message": f"达到最大步数 {max_steps}，任务未完成",
            "history_length": len(self.history),
        }

    # =========================
    # 工具方法
    # =========================
    def reset_history(self) -> None:
        self.history.clear()

    def get_history(self) -> list[AgentStep]:
        return self.history[:]