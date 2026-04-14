from pathlib import Path


DEFAULT_MODEL_NAME = "doubao-seed-2-0-lite-260215"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_AUDIT_REQUEST = "Run a static code audit for the current workspace, focusing on bugs, performance issues, and optimization opportunities."

DEFAULT_MAX_STEPS = 5
DEFAULT_MAX_REPLANS = 1
DEFAULT_GUI_MAX_STEPS = 8
MAX_TOOL_CALLS_PER_STEP = 8
MIN_EVIDENCE_PER_STEP = 1
ALLOW_REPLAN_REASONS = {"no_evidence", "low_confidence"}

PROMPT_FILE = Path(__file__).resolve().parent.parent / "PROMPT.md"
