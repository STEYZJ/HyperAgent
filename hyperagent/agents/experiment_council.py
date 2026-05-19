"""Multi-agent experiment council to avoid tunnel-vision search."""

from typing import Dict, List, Optional

from hyperagent.schemas import (
    DatasetAudit,
    ExperimentCouncilDecision,
    ExperimentCouncilVote,
    ExperimentCycle,
    ExperimentDiagnosis,
    ExperimentPlan,
    ParameterProposal,
)


class ExperimentCouncilAgent:
    """Role-based reviewers for autonomous experiment decisions."""

    def review(
        self,
        diagnosis: ExperimentDiagnosis,
        proposals: List[ParameterProposal],
        audit: DatasetAudit,
        plan: ExperimentPlan,
        history: List[ExperimentCycle],
        target_oa: float = 0.9,
        max_repeated_parameter: int = 2,
    ) -> ExperimentCouncilDecision:
        recent_counts = self._recent_parameter_counts(history)
        selected = proposals[0] if proposals else None
        rejected: List[str] = []
        checks: List[str] = []
        warnings: List[str] = []

        votes = [
            self._result_analyst_vote(diagnosis, target_oa),
            self._hypothesis_vote(selected),
            self._skeptic_vote(selected, recent_counts, max_repeated_parameter),
            self._reproducibility_vote(diagnosis, selected),
            self._budget_vote(history),
        ]

        if selected and recent_counts.get(selected.parameter, 0) >= max_repeated_parameter:
            rejected.append(selected.parameter)
            checks.append(
                f"Rejected repeated parameter `{selected.parameter}` after "
                f"{recent_counts[selected.parameter]} recent selections."
            )
            selected = self._first_non_repeated(
                proposals,
                recent_counts,
                max_repeated_parameter,
            )
            if selected is None:
                warnings.append(
                    "No non-repeated proposal is available; pause and request a new hypothesis or module/literature review."
                )

        if diagnosis.overall_accuracy >= target_oa and selected and selected.parameter != "seed":
            rejected.append(selected.parameter)
            checks.append(
                "High-accuracy run should verify seed stability before changing data/model parameters."
            )
            seed_proposal = self._find_parameter(proposals, "seed")
            selected = seed_proposal
            if seed_proposal is None:
                warnings.append("High-accuracy run has no seed-stability proposal.")

        if selected is None:
            action = "pause"
            rationale = "Council paused the cycle because no suitable non-redundant next experiment was available."
        else:
            action = "run"
            rationale = (
                f"Council selected `{selected.parameter}` because it is supported by evidence and passes anti-tunnel checks."
            )
        if history:
            checks.append(f"Reviewed {len(history)} prior cycle(s) for repeated search direction.")
        else:
            checks.append("No prior cycles found; no repetition risk from history.")
        checks.append(f"Dataset context: {audit.labeled_pixel_count} labeled pixels, {audit.class_count} classes.")
        checks.append(f"Current plan model={plan.model.name}, seed={plan.seed}.")

        return ExperimentCouncilDecision(
            action=action,
            selected_parameter=selected.parameter if selected else None,
            rationale=rationale,
            votes=votes,
            rejected_parameters=sorted(set(rejected)),
            anti_tunnel_checks=checks,
            warnings=warnings,
        )

    def _result_analyst_vote(
        self,
        diagnosis: ExperimentDiagnosis,
        target_oa: float,
    ) -> ExperimentCouncilVote:
        if diagnosis.overall_accuracy >= target_oa:
            decision = "seed_stability"
            rationale = (
                f"OA={diagnosis.overall_accuracy:.4f} meets target {target_oa:.4f}; test stability before changing assumptions."
            )
        else:
            decision = "targeted_change"
            rationale = (
                f"OA={diagnosis.overall_accuracy:.4f} is below target {target_oa:.4f}; continue with a targeted change."
            )
        return ExperimentCouncilVote(
            agent_name="ResultAnalystAgent",
            role="metric diagnosis",
            decision=decision,
            rationale=rationale,
            confidence=0.85,
        )

    def _hypothesis_vote(
        self,
        selected: Optional[ParameterProposal],
    ) -> ExperimentCouncilVote:
        if selected is None:
            return ExperimentCouncilVote(
                agent_name="HypothesisAgent",
                role="next-experiment hypothesis",
                decision="no_proposal",
                rationale="No parameter proposal was available.",
                confidence=0.4,
                warnings=["The experiment loop needs a new hypothesis source."],
            )
        return ExperimentCouncilVote(
            agent_name="HypothesisAgent",
            role="next-experiment hypothesis",
            decision=f"try_{selected.parameter}",
            rationale=selected.rationale,
            confidence=0.75,
        )

    def _skeptic_vote(
        self,
        selected: Optional[ParameterProposal],
        recent_counts: Dict[str, int],
        max_repeated_parameter: int,
    ) -> ExperimentCouncilVote:
        if selected is None:
            return ExperimentCouncilVote(
                agent_name="SkepticAgent",
                role="anti-tunnel guard",
                decision="pause",
                rationale="Cannot evaluate repetition risk without a selected proposal.",
                confidence=0.7,
            )
        count = recent_counts.get(selected.parameter, 0)
        if count >= max_repeated_parameter:
            return ExperimentCouncilVote(
                agent_name="SkepticAgent",
                role="anti-tunnel guard",
                decision="reject_repeated_parameter",
                rationale=(
                    f"`{selected.parameter}` was selected {count} time(s) recently; repeating it risks tunnel vision."
                ),
                confidence=0.9,
                warnings=[f"Repeated parameter: {selected.parameter}"],
            )
        return ExperimentCouncilVote(
            agent_name="SkepticAgent",
            role="anti-tunnel guard",
            decision="allow",
            rationale=f"`{selected.parameter}` is not overused in recent cycles.",
            confidence=0.8,
        )

    def _reproducibility_vote(
        self,
        diagnosis: ExperimentDiagnosis,
        selected: Optional[ParameterProposal],
    ) -> ExperimentCouncilVote:
        if diagnosis.overall_accuracy >= 0.9 and selected and selected.parameter == "seed":
            decision = "supports_seed_check"
            rationale = "High baseline accuracy should be tested across seeds before architectural changes."
        else:
            decision = "requires_artifact_record"
            rationale = "The next experiment must preserve plan/result/report artifacts for comparison."
        return ExperimentCouncilVote(
            agent_name="ReproducibilityAgent",
            role="reproducibility check",
            decision=decision,
            rationale=rationale,
            confidence=0.8,
        )

    def _budget_vote(self, history: List[ExperimentCycle]) -> ExperimentCouncilVote:
        if len(history) >= 8:
            return ExperimentCouncilVote(
                agent_name="BudgetAgent",
                role="budget and breadth control",
                decision="review_before_more_runs",
                rationale="Many cycles already exist; summarize evidence before launching more runs.",
                confidence=0.75,
                warnings=["Cycle history is getting long."],
            )
        return ExperimentCouncilVote(
            agent_name="BudgetAgent",
            role="budget and breadth control",
            decision="allow_one_more_run",
            rationale=f"{len(history)} prior cycle(s) found; budget allows another bounded run.",
            confidence=0.7,
        )

    def _recent_parameter_counts(self, history: List[ExperimentCycle], window: int = 5) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cycle in history[-window:]:
            proposal = cycle.selected_proposal
            if proposal is None:
                continue
            counts[proposal.parameter] = counts.get(proposal.parameter, 0) + 1
        return counts

    def _first_non_repeated(
        self,
        proposals: List[ParameterProposal],
        recent_counts: Dict[str, int],
        max_repeated_parameter: int,
    ) -> Optional[ParameterProposal]:
        for proposal in proposals:
            if recent_counts.get(proposal.parameter, 0) < max_repeated_parameter:
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
