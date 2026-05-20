"""Executable multi-agent experiment council runtime."""

from dataclasses import dataclass, field
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hyperagent.agents.experiment_council import ExperimentCouncilAgent
from hyperagent.runtime.deepseek_reasonix import get_reasonix_profile
from hyperagent.runtime.extensions import RuntimeExtensionStore
from hyperagent.runtime.llm import LLMClient, LLMProviderStore
from hyperagent.runtime.llm_usage import LLMUsageLedger
from hyperagent.runtime.workspace import utc_now
from hyperagent.schemas import (
    DatasetAudit,
    EvidenceItem,
    ExperimentCouncilDecision,
    ExperimentCouncilRoleRun,
    ExperimentCouncilRun,
    ExperimentCouncilVote,
    ExperimentCycle,
    ExperimentDiagnosis,
    ExperimentPlan,
    LLMMessage,
    ParameterProposal,
)


COUNCIL_ROLES = {
    "result_analyst": {
        "agent_name": "ResultAnalystAgent",
        "role": "metric diagnosis",
        "tools": ["read_result", "compare_metrics"],
    },
    "hypothesis": {
        "agent_name": "HypothesisAgent",
        "role": "next-experiment hypothesis",
        "tools": ["read_proposals", "inspect_evidence"],
    },
    "skeptic": {
        "agent_name": "SkepticAgent",
        "role": "anti-tunnel guard",
        "tools": ["inspect_history", "detect_repetition"],
    },
    "reproducibility": {
        "agent_name": "ReproducibilityAgent",
        "role": "reproducibility check",
        "tools": ["inspect_split", "inspect_artifacts"],
    },
    "budget": {
        "agent_name": "BudgetAgent",
        "role": "budget and breadth control",
        "tools": ["inspect_cycles", "inspect_budget"],
    },
}


@dataclass
class CouncilRoleConfig:
    role_key: str
    agent_name: str
    role: str
    tools: List[str] = field(default_factory=list)
    model: str = ""
    profile: str = ""


class ExecutableExperimentCouncilAgent:
    """Runs explicit role reviewers and coordinates a final decision."""

    def __init__(
        self,
        *,
        workspace_dir: Optional[Path] = None,
        llm_store: Optional[LLMProviderStore] = None,
        llm_client: Optional[LLMClient] = None,
        provider: str = "deepseek",
    ) -> None:
        self.static_council = ExperimentCouncilAgent()
        self.workspace_dir = workspace_dir
        self.llm_store = llm_store
        self.llm_client = llm_client or LLMClient()
        self.provider = provider

    def review(
        self,
        diagnosis: ExperimentDiagnosis,
        proposals: List[ParameterProposal],
        audit: DatasetAudit,
        plan: ExperimentPlan,
        history: List[ExperimentCycle],
        *,
        target_oa: float = 0.9,
        max_repeated_parameter: int = 2,
        llm_enabled: bool = False,
        llm_budget: int = 3,
        council_profile: str = "reasonix-balanced",
    ) -> ExperimentCouncilRun:
        selected = proposals[0] if proposals else None
        role_configs = self._load_role_configs(council_profile)
        role_runs: List[ExperimentCouncilRoleRun] = []
        warnings: List[str] = []
        budget_used = 0

        static_votes = {
            "result_analyst": self.static_council._result_analyst_vote(
                diagnosis,
                target_oa,
            ),
            "hypothesis": self.static_council._hypothesis_vote(selected),
            "skeptic": self.static_council._skeptic_vote(
                selected,
                self.static_council._recent_parameter_counts(history),
                max_repeated_parameter,
            ),
            "reproducibility": self.static_council._reproducibility_vote(
                diagnosis,
                selected,
            ),
            "budget": self.static_council._budget_vote(history),
        }

        for role_key in COUNCIL_ROLES:
            config = role_configs[role_key]
            base_vote = static_votes[role_key]
            role_run = self._role_run_from_vote(
                base_vote,
                config,
                diagnosis,
            )
            if llm_enabled and budget_used < max(llm_budget, 0):
                enhanced = self._try_llm_enhance(
                    role_run,
                    config,
                    diagnosis,
                    selected,
                    audit,
                    plan,
                    history,
                    council_profile,
                )
                role_run = enhanced
                if enhanced.llm_used:
                    budget_used += enhanced.budget_used
            elif llm_enabled and budget_used >= max(llm_budget, 0):
                role_run.warnings.append("LLM council budget exhausted before this role.")
            role_runs.append(role_run)

        final_decision = self._coordinate_decision(
            diagnosis,
            proposals,
            audit,
            plan,
            history,
            role_runs,
            target_oa=target_oa,
            max_repeated_parameter=max_repeated_parameter,
        )
        warnings.extend(final_decision.warnings)
        if llm_enabled and budget_used >= max(llm_budget, 0):
            warnings.append("LLM council budget exhausted.")

        return ExperimentCouncilRun(
            run_id=self._new_run_id(),
            mode="executable",
            llm_enabled=llm_enabled,
            budget_limit=max(llm_budget, 0),
            budget_used=budget_used,
            role_runs=role_runs,
            final_decision=final_decision,
            warnings=sorted(set(warnings)),
        )

    def _coordinate_decision(
        self,
        diagnosis: ExperimentDiagnosis,
        proposals: List[ParameterProposal],
        audit: DatasetAudit,
        plan: ExperimentPlan,
        history: List[ExperimentCycle],
        role_runs: List[ExperimentCouncilRoleRun],
        *,
        target_oa: float,
        max_repeated_parameter: int,
    ) -> ExperimentCouncilDecision:
        decision = self.static_council.review(
            diagnosis,
            proposals,
            audit,
            plan,
            history,
            target_oa=target_oa,
            max_repeated_parameter=max_repeated_parameter,
        )
        rejected = set(decision.rejected_parameters)
        checks = list(decision.anti_tunnel_checks)
        warnings = list(decision.warnings)

        repeated_direction = self._detect_repeated_direction(proposals, history)
        if repeated_direction:
            parameter = proposals[0].parameter if proposals else None
            if parameter:
                rejected.add(parameter)
            checks.append(repeated_direction)
            warnings.append("Repeated experiment direction detected.")
            alternative = self._first_allowed_proposal(proposals, rejected)
            if alternative is not None:
                decision.selected_parameter = alternative.parameter
                decision.rationale = (
                    f"Executable council rejected a repeated direction and selected "
                    f"`{alternative.parameter}` as the first non-repeated alternative."
                )
            else:
                decision.action = "pause"
                decision.selected_parameter = None
                decision.rationale = (
                    "Executable council paused because the available next step repeats "
                    "a recent experiment direction without new evidence."
                )

        if not proposals:
            warnings.append("No parameter proposals were available.")
        elif not any(proposal.evidence for proposal in proposals):
            checks.append("No proposal carried explicit evidence; require review before running.")
            warnings.append("No explicit proposal evidence found.")
            if len(history) > 0 and len(proposals) == 1:
                decision.action = "pause"
                decision.selected_parameter = None
                decision.rationale = (
                    "Executable council paused because a single unevidenced proposal "
                    "would continue the loop without a new basis."
                )

        if diagnosis.overall_accuracy >= target_oa and not self._find_parameter(proposals, "seed"):
            warnings.append("High-accuracy run has no seed-stability proposal.")
            decision.action = "pause"
            decision.selected_parameter = None
            decision.rationale = (
                "Executable council paused because high accuracy requires seed "
                "stability evidence before changing assumptions."
            )

        decision.votes = [
            ExperimentCouncilVote(
                agent_name=run.agent_name,
                role=run.role,
                decision=run.decision,
                rationale=run.rationale,
                confidence=run.confidence,
                warnings=list(run.warnings),
            )
            for run in role_runs
        ]
        decision.rejected_parameters = sorted(rejected)
        decision.anti_tunnel_checks = checks
        decision.warnings = sorted(set(warnings))
        return decision

    def _try_llm_enhance(
        self,
        role_run: ExperimentCouncilRoleRun,
        config: CouncilRoleConfig,
        diagnosis: ExperimentDiagnosis,
        selected: Optional[ParameterProposal],
        audit: DatasetAudit,
        plan: ExperimentPlan,
        history: List[ExperimentCycle],
        default_profile: str,
    ) -> ExperimentCouncilRoleRun:
        if self.llm_store is None:
            role_run.warnings.append("LLM council requested but no provider store is configured.")
            return role_run
        try:
            self.llm_store.ensure_defaults()
            spec = self.llm_store.get(self.provider)
            profile_name = config.profile or default_profile
            profile = get_reasonix_profile(profile_name)
            model = config.model or (profile.model if profile else None)
            response = self.llm_client.send(
                spec,
                self._llm_messages(
                    role_run,
                    diagnosis,
                    selected,
                    audit,
                    plan,
                    history,
                ),
                model=model,
                thinking={"type": profile.thinking} if profile and profile.thinking else None,
                reasoning_effort=profile.reasoning_effort if profile else None,
            )
            if self.workspace_dir is not None:
                LLMUsageLedger(self.workspace_dir).record_response(
                    response,
                    spec=spec,
                    event_type="experiment_council.role_vote",
                    metadata={"role": config.role_key, "profile": profile_name},
                )
            role_run.budget_used = 1
            role_run.model = model or spec.default_model
            role_run.profile = profile_name
            if response.warnings:
                role_run.warnings.extend(response.warnings)
                return role_run
            parsed = self._parse_llm_vote(response.content)
            if parsed:
                role_run.decision = str(parsed.get("decision", role_run.decision))
                role_run.rationale = str(parsed.get("rationale", role_run.rationale))
                role_run.confidence = float(parsed.get("confidence", role_run.confidence))
                role_run.warnings.extend(str(v) for v in parsed.get("warnings", []))
            role_run.llm_used = True
            return role_run
        except Exception as exc:  # pragma: no cover - defensive fallback
            role_run.warnings.append(f"LLM council fallback: {type(exc).__name__}: {exc}")
            return role_run

    def _llm_messages(
        self,
        role_run: ExperimentCouncilRoleRun,
        diagnosis: ExperimentDiagnosis,
        selected: Optional[ParameterProposal],
        audit: DatasetAudit,
        plan: ExperimentPlan,
        history: List[ExperimentCycle],
    ) -> List[LLMMessage]:
        proposal_text = (
            json.dumps(selected.to_dict(), ensure_ascii=False)
            if selected
            else "none"
        )
        prompt = {
            "role": role_run.role,
            "current_decision": role_run.decision,
            "current_rationale": role_run.rationale,
            "diagnosis": diagnosis.to_dict(),
            "selected_proposal": proposal_text,
            "dataset": {
                "name": audit.dataset_name,
                "classes": audit.class_count,
                "bands": audit.band_count,
                "labeled_pixels": audit.labeled_pixel_count,
            },
            "plan": {
                "experiment_name": plan.experiment_name,
                "model": plan.model.name,
                "seed": plan.seed,
            },
            "history_count": len(history),
        }
        return [
            LLMMessage(
                role="system",
                content=(
                    "You are one reviewer in HyperAgent's HSI experiment council. "
                    "Return one JSON object with decision, rationale, confidence, "
                    "and optional warnings. Do not request tool execution."
                ),
            ),
            LLMMessage(role="user", content=json.dumps(prompt, ensure_ascii=False)),
        ]

    def _parse_llm_vote(self, content: str) -> Dict[str, Any]:
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _role_run_from_vote(
        self,
        vote: ExperimentCouncilVote,
        config: CouncilRoleConfig,
        diagnosis: ExperimentDiagnosis,
    ) -> ExperimentCouncilRoleRun:
        evidence = list(diagnosis.evidence[:2])
        return ExperimentCouncilRoleRun(
            agent_name=config.agent_name,
            role=config.role,
            decision=vote.decision,
            rationale=vote.rationale,
            confidence=vote.confidence,
            evidence=evidence,
            warnings=list(vote.warnings),
            llm_used=False,
            budget_used=0,
            model=config.model,
            profile=config.profile,
            tools=list(config.tools),
        )

    def _load_role_configs(self, default_profile: str) -> Dict[str, CouncilRoleConfig]:
        configs = {
            key: CouncilRoleConfig(
                role_key=key,
                agent_name=str(value["agent_name"]),
                role=str(value["role"]),
                tools=[str(item) for item in value["tools"]],
                profile=default_profile,
            )
            for key, value in COUNCIL_ROLES.items()
        }
        if self.workspace_dir is None:
            return configs
        store = RuntimeExtensionStore(self.workspace_dir)
        for item in store.list_subagents():
            role_key = str(item.get("role", ""))
            if role_key not in configs:
                continue
            configs[role_key] = CouncilRoleConfig(
                role_key=role_key,
                agent_name=str(item.get("name") or configs[role_key].agent_name),
                role=role_key,
                tools=[str(v) for v in item.get("tools", [])],
                model=str(item.get("model", "")),
                profile=str(item.get("profile", "")) or default_profile,
            )
        return configs

    def _detect_repeated_direction(
        self,
        proposals: List[ParameterProposal],
        history: List[ExperimentCycle],
    ) -> str:
        if not proposals or not history:
            return ""
        selected = proposals[0]
        for cycle in history[-3:]:
            previous = cycle.selected_proposal
            if previous is None:
                continue
            if previous.parameter == selected.parameter:
                similarity = SequenceMatcher(
                    None,
                    str(previous.rationale).lower(),
                    str(selected.rationale).lower(),
                ).ratio()
                if similarity >= 0.72:
                    return (
                        f"Repeated direction detected for `{selected.parameter}` "
                        f"with rationale similarity {similarity:.2f}."
                    )
        return ""

    def _first_allowed_proposal(
        self,
        proposals: List[ParameterProposal],
        rejected: set,
    ) -> Optional[ParameterProposal]:
        for proposal in proposals:
            if proposal.parameter not in rejected:
                return proposal
        return None

    def _find_parameter(
        self,
        proposals: List[ParameterProposal],
        parameter: str,
    ) -> Optional[ParameterProposal]:
        for proposal in proposals:
            if proposal.parameter == parameter:
                return proposal
        return None

    def _new_run_id(self) -> str:
        return f"council-{utc_now().replace(':', '').replace('-', '')}-{uuid4().hex[:6]}"
