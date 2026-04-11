from dataclasses import asdict

from planner.types import PlanStep, StepMode


def verify_step_result(step_mode: StepMode, result_text: str) -> tuple[bool, str]:
    text = result_text.strip()
    if not text:
        return False, "empty_result"
    lowered = text.lower()

    if step_mode == "gui_test":
        failed_markers = (
            "gui tool error",
            '"status": "failed"',
            '"status":"failed"',
            '"status": "timeout"',
            '"status":"timeout"',
        )
        if any(marker in lowered for marker in failed_markers):
            return False, "gui_execution_failed"
    return True, "ok"


def build_replan_reason(
    objective: str,
    summary: str,
    completed_steps: list[PlanStep],
    failed_step: PlanStep,
    reason: str,
) -> str:
    completed = "\n\n".join(
        f"{step.id} | {step.title} | {step.status}\n目标: {step.objective}\n结果: {step.result}"
        for step in completed_steps
    ) or "无"
    failed_text = asdict(failed_step)
    return (
        f"总目标:\n{objective}\n\n"
        f"规划摘要:\n{summary}\n\n"
        f"已完成步骤:\n{completed}\n\n"
        f"失败原因:\n{reason}\n\n"
        f"失败步骤:\n{failed_text}"
    )
