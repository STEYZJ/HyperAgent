"""Reasonix-inspired parsing and repair helpers for agent tool calls."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hyperagent.schemas import LLMResponse


@dataclass
class ActionParseResult:
    action: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    source: str = "content"


class ToolCallStormBreaker:
    """Suppresses repeated identical tool calls inside one action run."""

    def __init__(self, max_repeats: int = 2) -> None:
        self.max_repeats = max(max_repeats, 1)
        self._counts: Dict[str, int] = {}

    def check(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        signature = json.dumps(
            {"tool_name": tool_name, "args": args},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        count = self._counts.get(signature, 0) + 1
        self._counts[signature] = count
        if count > self.max_repeats:
            return (
                f"Suppressed repeated tool call storm: {tool_name} repeated "
                f"{count} times with identical arguments."
            )
        return None


class ActionRepairPipeline:
    """Parse native tool calls, JSON content, and scavenged reasoning fragments."""

    TOOL_JSON_RE = re.compile(r"\{[^{}]*(?:\"action\"|\"tool_name\")[\s\S]*?\}", re.MULTILINE)

    def parse(self, response: LLMResponse) -> ActionParseResult:
        many = self.parse_many(response)
        return many[0] if many else ActionParseResult(
            action={"action": "final", "final": response.content},
            warnings=["Response was not valid JSON and was treated as final."],
            source="fallback",
        )

    def parse_many(self, response: LLMResponse) -> List[ActionParseResult]:
        native = self._parse_native_tool_calls(response.tool_calls)
        if native:
            return native

        content_result = self._parse_text(response.content, source="content")
        if content_result is not None:
            return [content_result]

        reasoning_result = self._parse_text(
            response.reasoning_content,
            source="reasoning_content",
        )
        if reasoning_result is not None:
            reasoning_result.warnings.append(
                "Tool call was scavenged from reasoning_content; ask the model to emit tool calls in content or native tool_calls."
            )
            return [reasoning_result]

        return [
            ActionParseResult(
                action={"action": "final", "final": response.content},
                warnings=["Response was not valid JSON and was treated as final."],
                source="fallback",
            )
        ]

    def _parse_native_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ActionParseResult]:
        if not tool_calls:
            return []
        results: List[ActionParseResult] = []
        for item in tool_calls:
            function = item.get("function", {}) if isinstance(item, dict) else {}
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            raw_args = function.get("arguments", "{}")
            args, warning = self._loads_object(str(raw_args))
            warnings = [warning] if warning else []
            results.append(
                ActionParseResult(
                    action={
                        "thought": "native tool call",
                        "action": "tool",
                        "tool_name": name,
                        "args": args,
                    },
                    warnings=warnings,
                    source="native_tool_calls",
                )
            )
        return results

    def _parse_text(self, text: str, *, source: str) -> Optional[ActionParseResult]:
        stripped = (text or "").strip()
        if not stripped:
            return None
        candidates = [self._strip_fence(stripped)]
        candidates.extend(self._scavenge_json_candidates(stripped))
        for candidate in candidates:
            parsed, warning = self._loads_object(candidate)
            if not parsed:
                continue
            if "action" not in parsed and "tool_name" in parsed:
                parsed = {"action": "tool", **parsed}
            if "action" in parsed:
                warnings = [warning] if warning else []
                return ActionParseResult(action=parsed, warnings=warnings, source=source)
        repaired = self._repair_truncated_json(stripped)
        if repaired:
            parsed, warning = self._loads_object(repaired)
            if parsed and "action" in parsed:
                warnings = ["Repaired truncated JSON action."]
                if warning:
                    warnings.append(warning)
                return ActionParseResult(action=parsed, warnings=warnings, source=source)
        return None

    def _loads_object(self, text: str) -> Tuple[Dict[str, Any], str]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return {}, f"JSON parse failed: {exc}"
        if not isinstance(parsed, dict):
            return {}, "JSON root was not an object."
        return parsed, ""

    def _strip_fence(self, text: str) -> str:
        lines = text.strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _scavenge_json_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        marker_patterns = [
            re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", re.IGNORECASE),
            re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE),
        ]
        for pattern in marker_patterns:
            candidates.extend(match.group(1).strip() for match in pattern.finditer(text))
        candidates.extend(match.group(0).strip() for match in self.TOOL_JSON_RE.finditer(text))
        return candidates

    def _repair_truncated_json(self, text: str) -> str:
        start = text.find("{")
        if start < 0:
            return ""
        candidate = text[start:].strip()
        candidate = self._strip_fence(candidate)
        depth = 0
        in_string = False
        escaped = False
        output: List[str] = []
        for char in candidate:
            output.append(char)
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
        if in_string:
            output.append('"')
        if depth > 0:
            output.extend("}" for _ in range(depth))
        repaired = "".join(output)
        return repaired if repaired != candidate else ""
