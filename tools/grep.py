import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from langchain.tools import tool
from file.ripgrep import Ripgrep
from tools.core import AsyncProcess, Filesystem, ProjectContext, WorkspaceGuard

# ======================== 4. Agent 集成的 GrepTool 核心类 ========================
class GrepToolParams(BaseModel):
    """参数校验模型：对应 grep.ts 的 z.object 定义"""
    pattern: str = Field(description="The regex pattern to search for in file contents")
    path: Optional[str] = Field(default=None, description="The directory to search in. Defaults to the current working directory.")
    include: Optional[str] = Field(default=None, description='File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")')

    @field_validator("pattern")
    def validate_pattern(cls, v: str) -> str:
        """校验搜索模式非空"""
        if not v.strip():
            raise ValueError("pattern is required (cannot be empty)")
        return v

class GrepTool:
    """
    Agent 集成的 Ripgrep 工具类：完全对齐 grep.ts 的核心逻辑
    可直接接入 Agent 的工具调用流程
    """
    NAME = "grep"
    DESCRIPTION = """Search file contents using ripgrep (rg) - a fast regex search tool.
Supports regex patterns, directory scoping, and file pattern inclusion.
Exit codes: 0 = matches found, 1 = no matches, 2 = partial errors (e.g. inaccessible paths)."""
    MAX_LINE_LENGTH = 2000  # 单行截断长度
    RESULT_LIMIT = 100       # 结果数量限制

    @classmethod
    async def execute(cls, params: GrepToolParams, ctx: Any = None) -> Dict[str, Any]:
        """
        核心执行逻辑（Agent 调用入口）
        :param params: 校验后的参数（pattern/path/include）
        :param ctx: Agent 上下文（需包含权限请求、abort 信号等）
        :return: 结构化结果（title/metadata/output）
        """
        # 1. 权限请求（模拟 ctx.ask，Agent 中替换为真实权限校验）
        await cls._request_permission(ctx, params)

        # 2. 路径处理：相对路径转绝对路径，校验合法性
        search_path = cls._resolve_search_path(params.path)
        WorkspaceGuard.ensure_under_workspace(ProjectContext.directory, search_path)

        # 3. 构造 rg 命令参数
        rg_path = await Ripgrep.filepath()
        rg_args = cls._build_rg_args(params, search_path)

        # 4. 执行 rg 进程
        proc, exit_future = await AsyncProcess.spawn(
            [rg_path] + rg_args,
            cwd=ProjectContext.directory,
            opts={
                "stdout": "pipe",
                "stderr": "pipe",
                "abort": getattr(ctx, "abort", None)  # Agent 中断信号
            }
        )

        # 5. 读取输出和退出码
        stdout = await AsyncProcess.read_stream(proc.stdout)
        stderr = await AsyncProcess.read_stream(proc.stderr)
        exit_code = await exit_future

        # 6. 退出码处理（对齐 grep.ts 逻辑）
        cls._handle_exit_code(exit_code, stdout, stderr)

        # 7. 解析并格式化结果
        matches = cls._parse_rg_output(stdout)
        result = cls._format_result(matches, params.pattern, exit_code == 2)

        return result

    @classmethod
    async def _request_permission(cls, ctx: Any, params: GrepToolParams):
        """模拟权限请求（Agent 中替换为真实的权限校验逻辑）"""
        if not ctx:
            return
        if hasattr(ctx, "ask"):
            await ctx.ask({
                "permission": "grep",
                "patterns": [params.pattern],
                "always": ["*"],
                "metadata": {
                    "pattern": params.pattern,
                    "path": params.path,
                    "include": params.include,
                },
            })

    @classmethod
    def _resolve_search_path(cls, path: Optional[str]) -> Path:
        """解析搜索路径：默认项目根目录，相对路径转绝对路径"""
        return ProjectContext.resolve_path(path)

    @classmethod
    def _build_rg_args(cls, params: GrepToolParams, search_path: Path) -> List[str]:
        """构造 rg 命令参数（对齐 grep.ts 的 args 逻辑）"""
        args = [
            "-nH",          # 显示行号+文件名
            "--hidden",     # 包含隐藏文件
            "--no-messages",# 隐藏错误消息
            "--field-match-separator=|",  # 字段分隔符
            "--regexp", params.pattern    # 搜索模式
        ]
        # 包含指定文件模式
        if params.include:
            args.extend(["--glob", params.include])
        # 搜索路径
        args.append(str(search_path))
        return args

    @classmethod
    def _handle_exit_code(cls, exit_code: int, stdout: str, stderr: str):
        """处理 rg 退出码（对齐 grep.ts 的错误逻辑）"""
        # 0: 有匹配；1: 无匹配；2: 错误（但可能有匹配）
        if exit_code == 1 or (exit_code == 2 and not stdout.strip()):
            return  # 无匹配/仅错误无输出，不抛异常
        if exit_code not in (0, 1, 2):
            raise RuntimeError(f"ripgrep failed (exit code {exit_code}): {stderr}")

    @classmethod
    def _parse_rg_output(cls, stdout: str) -> List[Dict[str, Any]]:
        """解析 rg 输出：拆分文件名/行号/内容，补充文件状态"""
        if not stdout:
            return []
        
        matches = []
        # 兼容 Unix(\n)/Windows(\r\n) 换行符
        lines = re.split(r"\r?\n", stdout.strip())
        for line in lines:
            if not line:
                continue
            # 按分隔符拆分：filePath|lineNum|lineText
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            file_path, line_num_str, line_text = parts
            # 校验行号
            try:
                line_num = int(line_num_str)
            except ValueError:
                continue
            # 获取文件修改时间
            stats = Filesystem.stat(file_path)
            if not stats:
                continue
            # 封装匹配结果
            matches.append({
                "path": file_path,
                "modTime": stats.st_mtime * 1000,  # 转毫秒（对齐 ts 的 getTime()）
                "lineNum": line_num,
                "lineText": line_text
            })
        # 按修改时间降序排序（最新文件优先）
        matches.sort(key=lambda x: x["modTime"], reverse=True)
        return matches

    @classmethod
    def _format_result(cls, matches: List[Dict[str, Any]], pattern: str, has_errors: bool) -> Dict[str, Any]:
        """格式化结果（对齐 grep.ts 的输出逻辑）"""
        # 无匹配
        if not matches:
            return {
                "title": pattern,
                "metadata": {"matches": 0, "truncated": False},
                "output": "No files found"
            }
        # 截断结果（超过100条仅显示前100）
        total_matches = len(matches)
        truncated = total_matches > cls.RESULT_LIMIT
        final_matches = matches[:cls.RESULT_LIMIT] if truncated else matches

        # 构造输出文本
        output_lines = [
            f"Found {total_matches} matches{f' (showing first {cls.RESULT_LIMIT})' if truncated else ''}"
        ]
        current_file = ""
        for match in final_matches:
            # 按文件分组
            if match["path"] != current_file:
                if current_file:
                    output_lines.append("")  # 文件间空行分隔
                current_file = match["path"]
                output_lines.append(f"{current_file}:")
            # 单行内容截断
            line_text = match["lineText"]
            if len(line_text) > cls.MAX_LINE_LENGTH:
                line_text = line_text[:cls.MAX_LINE_LENGTH] + "..."
            output_lines.append(f"  Line {match['lineNum']}: {line_text}")
        
        # 补充截断提示
        if truncated:
            output_lines.append("")
            output_lines.append(
                f"(Results truncated: showing {cls.RESULT_LIMIT} of {total_matches} matches. "
                "Consider using a more specific path or pattern.)"
            )
        # 补充错误提示（如部分路径不可访问）
        if has_errors:
            output_lines.append("")
            output_lines.append("(Some paths were inaccessible and skipped)")

        return {
            "title": pattern,
            "metadata": {"matches": total_matches, "truncated": truncated},
            "output": "\n".join(output_lines)
        }


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


@tool
def grep_search(pattern: str, path: Optional[str] = None, include: Optional[str] = None) -> str:
    """
    Search file contents using ripgrep and return formatted results.
    """
    try:
        params = GrepToolParams(pattern=pattern, path=path, include=include)
        result = _run_async(GrepTool.execute(params=params, ctx=None))
        return result["output"]
    except Exception as e:
        return f"Grep error: {str(e)}"

# ======================== 5. Agent 集成示例 ========================
async def agent_grep_demo():
    """Agent 中调用 GrepTool 的示例"""
    # 1. 构造 Agent 上下文（替换为真实上下文）
    class AgentContext:
        abort = asyncio.Event()  # 中断信号
        async def ask(self, permission: Dict):
            """模拟 Agent 权限请求"""
            print(f"[Agent Permission Check] {permission}")
            return True

    ctx = AgentContext()

    # 2. 构造搜索参数
    params = GrepToolParams(
        pattern=r"def\s+\w+",  # 搜索Python函数定义
        path="./",             # 搜索当前目录
        include="*.py"         # 仅包含py文件
    )

    # 3. 执行 grep 工具
    try:
        result = await GrepTool.execute(params, ctx)
        print(f"Title: {result['title']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Output:\n{result['output']}")
    except Exception as e:
        print(f"Grep failed: {e}")

