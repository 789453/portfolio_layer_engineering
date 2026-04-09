from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_project_tree(
    root_dir: str | Path,
    exclude_suffixes: list[str] | None = None,
) -> dict[str, Any]:
    """
    扫描 Python 项目目录结构，生成树形 JSON 数据。

    参数:
        root_dir: 项目根目录
        exclude_suffixes: 需要排除的文件后缀列表，例如 ['.log', '.md', '.txt']

    返回:
        一个可序列化为 JSON 的字典，包含目录和文件结构
    """
    root_path = Path(root_dir).resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"目录不存在: {root_path}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"不是目录: {root_path}")

    exclude_suffixes = exclude_suffixes or []
    exclude_suffixes = {suffix.lower() for suffix in exclude_suffixes}

    # 固定排除 .pyc
    exclude_suffixes.add(".pyc")

    def should_skip_file(path: Path) -> bool:
        if not path.is_file():
            return False
        return path.suffix.lower() in exclude_suffixes

    def should_skip_dir(path: Path) -> bool:
        return path.name == "__pycache__"

    def walk(path: Path) -> dict[str, Any] | None:
        if path.is_dir():
            if should_skip_dir(path):
                return None

            children: list[dict[str, Any]] = []

            try:
                entries = sorted(
                    path.iterdir(),
                    key=lambda p: (p.is_file(), p.name.lower())
                )
            except PermissionError:
                return {
                    "type": "directory",
                    "name": path.name,
                    "path": str(path.relative_to(root_path)) if path != root_path else ".",
                    "children": [],
                    "error": "Permission denied"
                }

            for entry in entries:
                node = walk(entry)
                if node is not None:
                    children.append(node)

            return {
                "type": "directory",
                "name": path.name if path != root_path else root_path.name,
                "path": str(path.relative_to(root_path)) if path != root_path else ".",
                "children": children,
            }

        if path.is_file():
            if should_skip_file(path):
                return None

            return {
                "type": "file",
                "name": path.name,
                "path": str(path.relative_to(root_path)),
                "suffix": path.suffix,
            }

        return None

    result = walk(root_path)
    if result is None:
        raise RuntimeError("生成项目树失败")

    return result


def save_project_tree_to_json(
    root_dir: str | Path,
    output_json: str | Path,
    exclude_suffixes: list[str] | None = None,
    indent: int = 2,
) -> None:
    """
    扫描项目目录并保存为 JSON 文件。

    参数:
        root_dir: 项目根目录
        output_json: 输出 JSON 文件路径
        exclude_suffixes: 需要排除的文件后缀列表
        indent: JSON 缩进
    """
    tree = build_project_tree(root_dir=root_dir, exclude_suffixes=exclude_suffixes)

    output_path = Path(output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=indent)


if __name__ == "__main__":
    # ===== 示例用法 =====
    project_root = r"D:\Trading\alpha_model_engineering"
    output_file = "3332project_structure.json"

    # 可手动排除的文件类型
    exclude_types = [
        ".log",
        ".tmp",
        ".csv",
        ".ipynb",
    ]

    save_project_tree_to_json(
        root_dir=project_root,
        output_json=output_file,
        exclude_suffixes=exclude_types,
    )

    print(f"项目结构已写入: {output_file}")
