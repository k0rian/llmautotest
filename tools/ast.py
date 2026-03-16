import ast
import inspect
from typing import List, Dict, Optional
from langchain_core.tools import tool

class CodeAnalyzer:
    """内部工具类：负责解析 Python 文件并提取结构信息"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, "r", encoding="utf-8") as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)

    def get_symbol_map(self) -> List[Dict]:
        """提取所有的类和函数定义及其位置"""
        symbols = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append({
                    "name": node.name,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno
                })
        return symbols

    def get_source_by_name(self, name: str) -> Optional[str]:
        """根据名称提取对应的代码片段"""
        lines = self.source.splitlines()
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                # 获取该定义在源码中的行范围
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
        return None

# --- 封装成 LangChain Tool ---

@tool
def list_file_symbols(file_path: str):
    """
    当你需要了解一个 Python 文件中有哪些类和函数时使用。
    它会返回名称、类型以及所在的行号范围。
    """
    try:
        analyzer = CodeAnalyzer(file_path)
        return analyzer.get_symbol_map()
    except Exception as e:
        return f"解析错误: {str(e)}"

@tool
def get_symbol_code(file_path: str, symbol_name: str):
    """
    当你需要查看某个特定函数或类的具体实现代码时使用。
    输入参数为文件路径和目标名称（函数名或类名）。
    """
    try:
        analyzer = CodeAnalyzer(file_path)
        code = analyzer.get_source_by_name(symbol_name)
        return code if code else f"未在 {file_path} 中找到 '{symbol_name}'"
    except Exception as e:
        return f"获取代码失败: {str(e)}"