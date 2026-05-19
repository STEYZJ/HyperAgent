"""Rule-based model recommender for HSI baselines."""

from typing import Any, Dict

from hyperagent.core.registries import model_recommender_registry
from hyperagent.schemas import (
    DatasetAudit,
    ModelCandidate,
    ModelRecommendation,
    SpectralReport,
)


class BasicModelRecommender:
    name = "basic"

    def recommend(
        self,
        audit: DatasetAudit,
        spectral_report: SpectralReport,
        constraints: Dict[str, Any],
    ) -> ModelRecommendation:
        labeled = audit.labeled_pixel_count
        bands = max(1, audit.band_count - len(spectral_report.recommended_removed_bands))
        class_count = max(1, audit.class_count)

        svm_score = 0.72
        mlp_score = 0.58
        if labeled < 2000:
            svm_score += 0.12
            mlp_score -= 0.05
        if labeled >= 4000:
            mlp_score += 0.18
        if bands > 128:
            svm_score -= 0.06
            mlp_score += 0.04
        if class_count > 12:
            mlp_score += 0.05

        candidates = [
            ModelCandidate(
                name="svm",
                score=round(svm_score, 3),
                rationale="Strong small-sample spectral baseline with low engineering risk.",
                params={"kernel": "rbf", "C": 10.0, "gamma": "scale"},
            ),
            ModelCandidate(
                name="mlp",
                score=round(mlp_score, 3),
                rationale="Lightweight neural baseline for larger labeled sets.",
                params={"hidden_dim": 64, "epochs": 30, "lr": 0.001, "batch_size": 128},
            ),
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        notes = [
            "Recommendation is limited to MVP-supported models: svm and mlp.",
            f"Effective band count after recommended removal: {bands}.",
        ]
        if constraints:
            notes.append(f"Constraints considered: {constraints}.")

        return ModelRecommendation(
            recommended_model=candidates[0].name,
            candidates=candidates,
            constraints=dict(constraints),
            notes=notes,
        )


model_recommender_registry.register(BasicModelRecommender.name, BasicModelRecommender(), replace=True)
