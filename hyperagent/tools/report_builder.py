"""Markdown report writer."""

from pathlib import Path

from hyperagent.schemas import ExperimentResult


class MarkdownReportBuilder:
    name = "markdown"

    def write(self, result: ExperimentResult, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation = result.evaluation
        lines = [
            f"# HyperAgent Report: {result.experiment_name}",
            "",
            "## Summary",
            "",
            f"- Status: {result.status}",
            f"- Model: {result.model_name}",
            f"- Seed: {result.seed}",
            f"- Train samples: {result.train_samples}",
            f"- Test samples: {result.test_samples}",
            f"- Duration seconds: {result.duration_sec:.3f}",
            "",
            "## Metrics",
            "",
            f"- OA: {evaluation.overall_accuracy:.4f}",
            f"- AA: {evaluation.average_accuracy:.4f}",
            f"- Kappa: {evaluation.kappa:.4f}",
            "",
            "## Per-class Accuracy",
            "",
        ]
        for label, value in sorted(evaluation.per_class_accuracy.items(), key=lambda item: int(item[0])):
            lines.append(f"- Class {label}: {value:.4f}")
        lines.extend(["", "## Confusion Matrix", "", "```text"])
        lines.extend(" ".join(str(v) for v in row) for row in evaluation.confusion_matrix)
        lines.extend(["```", "", "## Artifacts", ""])
        for artifact in result.artifacts:
            lines.append(f"- {artifact}")
        if result.warnings:
            lines.extend(["", "## Warnings", ""])
            for warning in result.warnings:
                lines.append(f"- {warning}")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

