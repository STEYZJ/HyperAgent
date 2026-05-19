"""Append-only worklog utility used by CLI workflows."""

from datetime import datetime
from pathlib import Path
from typing import Optional


def default_worklog_path(root: Path = Path(".")) -> Path:
    date_text = datetime.now().strftime("%Y-%m-%d")
    return root / "logs" / "worklog" / f"{date_text}.md"


def append_worklog(
    title: str,
    previous: str,
    current: str,
    rationale: str,
    action: str,
    effect: str,
    next_step: str,
    *,
    path: Optional[Path] = None,
) -> Path:
    target = path or default_worklog_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        step_number = existing.count("\n## Step ") + 1
    else:
        target.write_text(
            f"# HyperAgent Worklog - {datetime.now().strftime('%Y-%m-%d')}\n",
            encoding="utf-8",
        )
        step_number = 1
    entry = (
        f"\n## Step {step_number} - {title}\n"
        f"- 上一步：{previous}\n"
        f"- 这一步：{current}\n"
        f"- 为什么这么干：{rationale}\n"
        f"- 执行内容：{action}\n"
        f"- 效果：{effect}\n"
        f"- 下一步：{next_step}\n"
    )
    with target.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return target

