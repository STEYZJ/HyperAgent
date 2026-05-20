"""DeepSeek Reasonix-inspired runtime profiles and cache guidance."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ReasonixProfile:
    """Model/runtime preset for DeepSeek-backed research agent work."""

    name: str
    model: str
    thinking: Optional[str]
    reasoning_effort: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    intent: str
    use_cases: List[str] = field(default_factory=list)
    cache_policy: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


REASONIX_PROFILES: Dict[str, ReasonixProfile] = {
    "reasonix-cheap": ReasonixProfile(
        name="reasonix-cheap",
        model="deepseek-v4-flash",
        thinking="disabled",
        reasoning_effort=None,
        temperature=0.2,
        top_p=0.9,
        intent="Fast low-cost turns for search, summaries, report cleanup, and routing.",
        use_cases=[
            "literature query refinement",
            "experiment report summarization",
            "conversation title or lightweight triage",
        ],
        cache_policy=[
            "Keep the stable system prompt before dynamic user text.",
            "Avoid changing project memory inside the same cached session unless required.",
        ],
    ),
    "reasonix-balanced": ReasonixProfile(
        name="reasonix-balanced",
        model="deepseek-v4-pro",
        thinking="enabled",
        reasoning_effort="high",
        temperature=None,
        top_p=None,
        intent="Default research-copilot mode for experiment analysis and code planning.",
        use_cases=[
            "analyze HSI experiment results",
            "choose purposeful parameter changes",
            "design ablations from evidence",
        ],
        cache_policy=[
            "Put stable project rules, dataset cards, and tool schemas first.",
            "Append volatile experiment output and user requests last.",
        ],
    ),
    "reasonix-deep": ReasonixProfile(
        name="reasonix-deep",
        model="deepseek-v4-pro",
        thinking="enabled",
        reasoning_effort="max",
        temperature=None,
        top_p=None,
        intent="Heavy reasoning for algorithm design, failure diagnosis, and paper-to-module mapping.",
        use_cases=[
            "turn a paper idea into module proposals",
            "diagnose repeated failed experiments",
            "review multi-agent experiment decisions",
        ],
        cache_policy=[
            "Freeze the long knowledge prefix before the agent debate.",
            "Record assumptions and evidence IDs so later turns can reuse the same prefix.",
        ],
    ),
}


def list_reasonix_profiles() -> List[ReasonixProfile]:
    return [REASONIX_PROFILES[name] for name in sorted(REASONIX_PROFILES)]


def get_reasonix_profile(name: Optional[str]) -> Optional[ReasonixProfile]:
    if not name:
        return None
    try:
        return REASONIX_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(REASONIX_PROFILES))
        raise ValueError(f"Unknown Reasonix profile: {name}. Available: {available}") from exc


def reasonix_cache_guidance() -> Dict[str, Any]:
    """Return cache-first prompt partitioning guidance without vendor coupling."""

    return {
        "stable_prefix": [
            "system prompt",
            "project memory",
            "dataset cards",
            "spectral rules",
            "tool schemas",
        ],
        "semi_stable_context": [
            "task description",
            "benchmark protocol",
            "accepted literature notes",
            "current experiment plan",
        ],
        "volatile_suffix": [
            "latest user request",
            "fresh command output",
            "recent failed tool result",
            "temporary /btw question",
        ],
        "rule": (
            "Preserve the order and bytes of stable_prefix whenever possible; only "
            "mutate volatile_suffix between adjacent turns so provider-side prefix "
            "cache has a chance to hit."
        ),
    }
