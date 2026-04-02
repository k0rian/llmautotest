import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from llm_utils.config_loader import load_api_key
from tools.description.description import TOOL_DESCRIPTIONS
from tools.tools import build_tools


StepMode = Literal["code_audit", "gui_test", "analysis"]


@dataclass
class PlanStep:
    id: str
    title: str
    mode: StepMode
    objective: str
    expected_output: str
    status: str = "pending"
    result: str = ""


@dataclass
class PlannerRunResult:
    objective: str
    summary: str
    steps: list[PlanStep]
    final_response: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "summary": self.summary,
            "steps": [asdict(step) for step in self.steps],
            "final_response": self.final_response,
            "status": self.status,
        }


def build_planner_model(
    model_name: str = "doubao-seed-1-6-251015",
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=load_api_key(),
    )


def build_executor_agent(model: ChatOpenAI | None = None, verbose: bool = False):
    active_model = model or build_planner_model()
    return create_agent(
        active_model,
        tools=build_tools(),
        verbose=verbose,
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(text if isinstance(text, str) else json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _extract_text(payload: Any) -> str:
    if isinstance(payload, dict):
        if "messages" in payload and payload["messages"]:
            return _extract_text(payload["messages"][-1])
        if "output" in payload:
            return _extract_text(payload["output"])
        return _stringify(payload)
    content = getattr(payload, "content", None)
    if content is not None:
        return _stringify(content)
    return _stringify(payload)


def _parse_json_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
        if match:
            raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _tool_description_text() -> str:
    blocks: list[str] = []
    for name, meta in TOOL_DESCRIPTIONS.items():
        category = meta.get("category", "general")
        description = meta.get("description", "")
        blocks.append(f"{name} [{category}]\n{description}")
    return "\n\n".join(blocks)


class PlanAndExecutePlanner:
    def __init__(
        self,
        planner_model: ChatOpenAI | None = None,
        executor_agent: Any | None = None,
        gui_executor: Any | None = None,
        verbose: bool = False,
        max_replans: int = 1,
        gui_max_steps: int = 8,
    ) -> None:
        self.planner_model = planner_model or build_planner_model()
        self.executor_agent = executor_agent or build_executor_agent(self.planner_model, verbose=verbose)
        self.gui_executor = gui_executor
        self.max_replans = max(0, int(max_replans))
        self.gui_max_steps = max(1, int(gui_max_steps))

    def invoke(self, inputs: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(inputs, str):
            result = self.execute(inputs)
            return result.to_dict()
        objective = (
            inputs.get("objective")
            or inputs.get("task")
            or inputs.get("input")
            or inputs.get("query")
            or ""
        )
        context = _stringify(inputs.get("context", ""))
        max_steps = inputs.get("max_steps")
        result = self.execute(objective=objective, context=context, max_steps=max_steps)
        return result.to_dict()

    def create_plan(self, objective: str, context: str = "") -> tuple[str, list[PlanStep]]:
        prompt = (
            "你是一个面向代码审计与 GUI 功能测试的任务规划器。"
            "请根据用户目标输出一个可执行计划。"
            "步骤数量控制在 1 到 6 步，每步必须足够原子。"
            "mode 只能是 code_audit、gui_test、analysis 之一。"
            "如果任务涉及代码阅读、静态分析、LSP、文件搜索，优先使用 code_audit。"
            "如果任务涉及界面点击、输入、截图验证，使用 gui_test。"
            "如果任务是汇总、判断、整合结论，使用 analysis。"
            "只返回 JSON，不要使用 markdown。"
            'JSON 格式为 {"summary":"...","steps":[{"title":"...","mode":"code_audit","objective":"...","expected_output":"..."}]}。'
            f"\n\n可用工具摘要：\n{_tool_description_text()}"
            f"\n\n用户目标：\n{objective}"
            f"\n\n补充上下文：\n{context or '无'}"
        )
        raw = _extract_text(self.planner_model.invoke(prompt))
        try:
            payload = _parse_json_payload(raw)
            summary = _stringify(payload.get("summary", "")).strip() or "执行任务规划"
            steps_data = payload.get("steps", [])
            steps = self._normalize_steps(steps_data)
            if not steps:
                raise ValueError("planner returned empty steps")
            return summary, steps
        except Exception:
            fallback = [
                PlanStep(
                    id="step-1",
                    title="执行任务",
                    mode="code_audit",
                    objective=objective,
                    expected_output="返回与目标直接相关的执行结果",
                )
            ]
            return "执行任务规划", fallback

    def execute(self, objective: str, context: str = "", max_steps: int | None = None) -> PlannerRunResult:
        if not objective or not objective.strip():
            return PlannerRunResult(
                objective="",
                summary="",
                steps=[],
                final_response="objective 不能为空",
                status="failed",
            )
        summary, steps = self.create_plan(objective=objective.strip(), context=context)
        if max_steps is not None:
            limit = max(1, int(max_steps))
            steps = steps[:limit]
        replans = 0
        index = 0
        while index < len(steps):
            step = steps[index]
            ok, result = self.execute_step(objective=objective, context=context, step=step, history=steps[:index])
            step.result = result
            step.status = "completed" if ok else "failed"
            if ok:
                index += 1
                continue
            if replans >= self.max_replans:
                final_response = self.summarize_execution(objective, summary, steps)
                return PlannerRunResult(
                    objective=objective,
                    summary=summary,
                    steps=steps,
                    final_response=final_response,
                    status="failed",
                )
            replans += 1
            remaining = self.replan(objective=objective, context=context, summary=summary, completed_steps=steps[:index], failed_step=step)
            if not remaining:
                final_response = self.summarize_execution(objective, summary, steps)
                return PlannerRunResult(
                    objective=objective,
                    summary=summary,
                    steps=steps,
                    final_response=final_response,
                    status="failed",
                )
            steps = steps[:index] + remaining
        final_response = self.summarize_execution(objective, summary, steps)
        return PlannerRunResult(
            objective=objective,
            summary=summary,
            steps=steps,
            final_response=final_response,
            status="completed",
        )

    def execute_step(
        self,
        objective: str,
        context: str,
        step: PlanStep,
        history: list[PlanStep],
    ) -> tuple[bool, str]:
        try:
            if step.mode == "gui_test":
                return self._execute_gui_step(step)
            prompt = self._build_execution_prompt(
                objective=objective,
                context=context,
                step=step,
                history=history,
            )
            response = self.executor_agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                }
            )
            return True, _extract_text(response).strip()
        except Exception as exc:
            return False, str(exc)

    def replan(
        self,
        objective: str,
        context: str,
        summary: str,
        completed_steps: list[PlanStep],
        failed_step: PlanStep,
    ) -> list[PlanStep]:
        completed_text = self._history_text(completed_steps)
        failed_text = json.dumps(asdict(failed_step), ensure_ascii=False, indent=2)
        prompt = (
            "你需要基于已经完成的步骤和失败信息重新规划剩余步骤。"
            "不要重复已完成步骤，只输出剩余步骤。"
            "mode 只能是 code_audit、gui_test、analysis。"
            "步骤数量控制在 1 到 4 步。"
            "只返回 JSON，不要使用 markdown。"
            'JSON 格式为 {"steps":[{"title":"...","mode":"code_audit","objective":"...","expected_output":"..."}]}。'
            f"\n\n总目标：\n{objective}"
            f"\n\n规划摘要：\n{summary}"
            f"\n\n上下文：\n{context or '无'}"
            f"\n\n已完成步骤：\n{completed_text or '无'}"
            f"\n\n失败步骤：\n{failed_text}"
        )
        try:
            raw = _extract_text(self.planner_model.invoke(prompt))
            payload = _parse_json_payload(raw)
            return self._normalize_steps(payload.get("steps", []))
        except Exception:
            return []

    def summarize_execution(self, objective: str, summary: str, steps: list[PlanStep]) -> str:
        execution_text = "\n\n".join(
            json.dumps(asdict(step), ensure_ascii=False, indent=2) for step in steps
        )
        prompt = (
            "请根据规划与执行记录，生成最终交付结论。"
            "输出使用中文，包含任务结论、关键证据、未完成风险、下一步建议。"
            f"\n\n总目标：\n{objective}"
            f"\n\n规划摘要：\n{summary}"
            f"\n\n执行记录：\n{execution_text or '无'}"
        )
        try:
            return _extract_text(self.planner_model.invoke(prompt)).strip()
        except Exception:
            return execution_text

    def _normalize_steps(self, steps_data: Any) -> list[PlanStep]:
        steps: list[PlanStep] = []
        if not isinstance(steps_data, list):
            return steps
        for idx, item in enumerate(steps_data, start=1):
            if not isinstance(item, dict):
                continue
            mode = _stringify(item.get("mode", "code_audit")).strip() or "code_audit"
            if mode not in {"code_audit", "gui_test", "analysis"}:
                mode = "code_audit"
            steps.append(
                PlanStep(
                    id=f"step-{idx}",
                    title=_stringify(item.get("title", f"步骤 {idx}")).strip() or f"步骤 {idx}",
                    mode=mode,
                    objective=_stringify(item.get("objective", "")).strip() or _stringify(item.get("title", f"步骤 {idx}")),
                    expected_output=_stringify(item.get("expected_output", "")).strip() or "返回本步骤结果",
                )
            )
        return steps

    def _build_execution_prompt(
        self,
        objective: str,
        context: str,
        step: PlanStep,
        history: list[PlanStep],
    ) -> str:
        return (
            "你是执行代理，需要完成当前规划步骤。"
            "优先使用已有工具获取证据，不要空想。"
            "如果当前步骤无法完成，请明确说明阻塞点。"
            "输出使用中文。"
            f"\n\n总目标：\n{objective}"
            f"\n\n补充上下文：\n{context or '无'}"
            f"\n\n已完成步骤：\n{self._history_text(history) or '无'}"
            f"\n\n当前步骤标题：\n{step.title}"
            f"\n\n当前步骤模式：\n{step.mode}"
            f"\n\n当前步骤目标：\n{step.objective}"
            f"\n\n期望输出：\n{step.expected_output}"
            "\n\n请给出：\n1. 本步执行结果\n2. 关键证据\n3. 对总任务的影响"
        )

    def _history_text(self, history: list[PlanStep]) -> str:
        if not history:
            return ""
        return "\n\n".join(
            f"{step.id} | {step.title} | {step.status}\n目标: {step.objective}\n结果: {step.result}"
            for step in history
        )

    def _execute_gui_step(self, step: PlanStep) -> tuple[bool, str]:
        executor = self._get_gui_executor()
        payload = executor.run(step.objective, max_steps=self.gui_max_steps)
        result = json.dumps(payload, ensure_ascii=False, indent=2)
        status = _stringify(payload.get("status", "")).strip().lower() if isinstance(payload, dict) else ""
        return status == "completed", result

    def _get_gui_executor(self) -> Any:
        if self.gui_executor is not None:
            return self.gui_executor
        from gui_agent.agent import GUIAgent

        self.gui_executor = GUIAgent()
        return self.gui_executor


def build_plan_and_execute_planner(
    planner_model: ChatOpenAI | None = None,
    executor_agent: Any | None = None,
    gui_executor: Any | None = None,
    verbose: bool = False,
    max_replans: int = 1,
    gui_max_steps: int = 8,
) -> PlanAndExecutePlanner:
    return PlanAndExecutePlanner(
        planner_model=planner_model,
        executor_agent=executor_agent,
        gui_executor=gui_executor,
        verbose=verbose,
        max_replans=max_replans,
        gui_max_steps=gui_max_steps,
    )
