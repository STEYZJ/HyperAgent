"""Append-only worklog utility used by CLI workflows."""

from datetime import datetime
from pathlib import Path
import re
from typing import Optional


SECRET_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}")
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[A-Z0-9_]*\s*[:=]\s*)(['\"]?)([^\s,'\"\]}]+)(['\"]?)"
)


def default_worklog_path(root: Path = Path(".")) -> Path:
    date_text = datetime.now().strftime("%Y-%m-%d")
    return root / "logs" / "worklog" / f"{date_text}.md"


def redact_secrets(text: object) -> str:
    """Redact obvious secrets before they reach persistent worklogs."""

    value = str(text)
    value = SECRET_TOKEN_PATTERN.sub("[REDACTED_SECRET]", value)
    return SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED_SECRET]{match.group(4)}",
        value,
    )


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
        f"\n## Step {step_number} - {redact_secrets(title)}\n"
        f"- 上一步：{redact_secrets(previous)}\n"
        f"- 这一步：{redact_secrets(current)}\n"
        f"- 为什么这么干：{redact_secrets(rationale)}\n"
        f"- 执行内容：{redact_secrets(action)}\n"
        f"- 效果：{redact_secrets(effect)}\n"
        f"- 下一步：{redact_secrets(next_step)}\n"
    )
    with target.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return target
