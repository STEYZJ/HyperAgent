"""Research decision schemas for autonomous HSI experimentation."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    source_type: str
    source_id: str
    claim: str
    support: str
    confidence: float = 0.5
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        return cls(
            source_type=str(data["source_type"]),
            source_id=str(data["source_id"]),
            claim=str(data["claim"]),
            support=str(data["support"]),
            confidence=float(data.get("confidence", 0.5)),
            url=data.get("url"),
        )


@dataclass
class DecisionRecord:
    decision_type: str
    target: str
    choice: str
    rationale: str
    evidence: List[EvidenceItem] = field(default_factory=list)
    expected_effect: str = ""
    risk: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        return cls(
            decision_type=str(data["decision_type"]),
            target=str(data["target"]),
            choice=str(data["choice"]),
            rationale=str(data["rationale"]),
            evidence=[
                EvidenceItem.from_dict(item) for item in data.get("evidence", [])
            ],
            expected_effect=str(data.get("expected_effect", "")),
            risk=str(data.get("risk", "")),
        )


@dataclass
class ExperimentCandidate:
    name: str
    plan_patch: Dict[str, Any]
    priority: float
    objective: str
    decisions: List[DecisionRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentCandidate":
        return cls(
            name=str(data["name"]),
            plan_patch=dict(data.get("plan_patch", {})),
            priority=float(data.get("priority", 0.5)),
            objective=str(data.get("objective", "")),
            decisions=[
                DecisionRecord.from_dict(item) for item in data.get("decisions", [])
            ],
        )


@dataclass
class AutoExperimentAgenda:
    dataset_name: str
    objective: str
    candidates: List[ExperimentCandidate]
    stop_rules: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoExperimentAgenda":
        return cls(
            dataset_name=str(data["dataset_name"]),
            objective=str(data["objective"]),
            candidates=[
                ExperimentCandidate.from_dict(item)
                for item in data.get("candidates", [])
            ],
            stop_rules=[str(v) for v in data.get("stop_rules", [])],
            notes=[str(v) for v in data.get("notes", [])],
        )


@dataclass
class ParameterProposal:
    parameter: str
    old_value: Any
    new_value: Any
    rationale: str
    expected_effect: str
    evidence: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterProposal":
        return cls(
            parameter=str(data["parameter"]),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            rationale=str(data["rationale"]),
            expected_effect=str(data["expected_effect"]),
            evidence=[
                EvidenceItem.from_dict(item) for item in data.get("evidence", [])
            ],
        )


@dataclass
class ModuleProposal:
    name: str
    module_type: str
    insertion_point: str
    design_summary: str
    expected_effect: str
    implementation_steps: List[str]
    required_interfaces: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleProposal":
        return cls(
            name=str(data["name"]),
            module_type=str(data["module_type"]),
            insertion_point=str(data["insertion_point"]),
            design_summary=str(data["design_summary"]),
            expected_effect=str(data["expected_effect"]),
            implementation_steps=[str(v) for v in data.get("implementation_steps", [])],
            required_interfaces=[str(v) for v in data.get("required_interfaces", [])],
            evidence=[
                EvidenceItem.from_dict(item) for item in data.get("evidence", [])
            ],
            risks=[str(v) for v in data.get("risks", [])],
        )


@dataclass
class AblationVariant:
    name: str
    config_path: str
    purpose: str
    changed_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationStudy:
    name: str
    base_plan: str
    variants: List[AblationVariant]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
