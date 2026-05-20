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
class ExperimentDiagnosis:
    experiment_name: str
    objective: str
    overall_accuracy: float
    average_accuracy: float
    kappa: float
    weakest_classes: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    recommendation: str = ""
    should_continue: bool = True
    evidence: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentDiagnosis":
        return cls(
            experiment_name=str(data["experiment_name"]),
            objective=str(data["objective"]),
            overall_accuracy=float(data.get("overall_accuracy", 0.0)),
            average_accuracy=float(data.get("average_accuracy", 0.0)),
            kappa=float(data.get("kappa", 0.0)),
            weakest_classes=[
                dict(item) for item in data.get("weakest_classes", [])
            ],
            findings=[str(v) for v in data.get("findings", [])],
            recommendation=str(data.get("recommendation", "")),
            should_continue=bool(data.get("should_continue", True)),
            evidence=[
                EvidenceItem.from_dict(item) for item in data.get("evidence", [])
            ],
        )


@dataclass
class ExperimentCouncilVote:
    agent_name: str
    role: str
    decision: str
    rationale: str
    confidence: float = 0.5
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentCouncilVote":
        return cls(
            agent_name=str(data["agent_name"]),
            role=str(data["role"]),
            decision=str(data["decision"]),
            rationale=str(data["rationale"]),
            confidence=float(data.get("confidence", 0.5)),
            warnings=[str(v) for v in data.get("warnings", [])],
        )


@dataclass
class ExperimentCouncilDecision:
    action: str
    selected_parameter: Optional[str]
    rationale: str
    votes: List[ExperimentCouncilVote] = field(default_factory=list)
    rejected_parameters: List[str] = field(default_factory=list)
    anti_tunnel_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentCouncilDecision":
        return cls(
            action=str(data["action"]),
            selected_parameter=(
                None
                if data.get("selected_parameter") is None
                else str(data.get("selected_parameter"))
            ),
            rationale=str(data["rationale"]),
            votes=[
                ExperimentCouncilVote.from_dict(item)
                for item in data.get("votes", [])
            ],
            rejected_parameters=[str(v) for v in data.get("rejected_parameters", [])],
            anti_tunnel_checks=[str(v) for v in data.get("anti_tunnel_checks", [])],
            warnings=[str(v) for v in data.get("warnings", [])],
        )


@dataclass
class ExperimentCouncilRoleRun:
    agent_name: str
    role: str
    decision: str
    rationale: str
    confidence: float = 0.5
    evidence: List[EvidenceItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    llm_used: bool = False
    budget_used: int = 0
    model: str = ""
    profile: str = ""
    tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentCouncilRoleRun":
        return cls(
            agent_name=str(data["agent_name"]),
            role=str(data["role"]),
            decision=str(data["decision"]),
            rationale=str(data["rationale"]),
            confidence=float(data.get("confidence", 0.5)),
            evidence=[
                EvidenceItem.from_dict(item) for item in data.get("evidence", [])
            ],
            warnings=[str(v) for v in data.get("warnings", [])],
            llm_used=bool(data.get("llm_used", False)),
            budget_used=int(data.get("budget_used", 0)),
            model=str(data.get("model", "")),
            profile=str(data.get("profile", "")),
            tools=[str(v) for v in data.get("tools", [])],
        )


@dataclass
class ExperimentCouncilRun:
    run_id: str
    mode: str
    llm_enabled: bool
    budget_limit: int
    budget_used: int
    role_runs: List[ExperimentCouncilRoleRun]
    final_decision: ExperimentCouncilDecision
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentCouncilRun":
        return cls(
            run_id=str(data["run_id"]),
            mode=str(data["mode"]),
            llm_enabled=bool(data.get("llm_enabled", False)),
            budget_limit=int(data.get("budget_limit", 0)),
            budget_used=int(data.get("budget_used", 0)),
            role_runs=[
                ExperimentCouncilRoleRun.from_dict(item)
                for item in data.get("role_runs", [])
            ],
            final_decision=ExperimentCouncilDecision.from_dict(
                data.get("final_decision", {})
            ),
            warnings=[str(v) for v in data.get("warnings", [])],
        )


@dataclass
class ExperimentCycle:
    cycle_id: str
    created_at: str
    status: str
    previous_plan_path: str
    previous_result_path: str
    audit_path: str
    cycle_dir: str
    diagnosis_path: str
    proposals_path: str
    next_plan_path: str
    council_path: Optional[str] = None
    council_run_path: Optional[str] = None
    selected_proposal: Optional[ParameterProposal] = None
    next_result_path: Optional[str] = None
    report_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentCycle":
        proposal = data.get("selected_proposal")
        return cls(
            cycle_id=str(data["cycle_id"]),
            created_at=str(data["created_at"]),
            status=str(data["status"]),
            previous_plan_path=str(data["previous_plan_path"]),
            previous_result_path=str(data["previous_result_path"]),
            audit_path=str(data["audit_path"]),
            cycle_dir=str(data["cycle_dir"]),
            diagnosis_path=str(data["diagnosis_path"]),
            proposals_path=str(data["proposals_path"]),
            next_plan_path=str(data["next_plan_path"]),
            council_path=data.get("council_path"),
            council_run_path=data.get("council_run_path"),
            selected_proposal=(
                None
                if proposal is None
                else ParameterProposal.from_dict(dict(proposal))
            ),
            next_result_path=data.get("next_result_path"),
            report_path=data.get("report_path"),
            warnings=[str(v) for v in data.get("warnings", [])],
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
