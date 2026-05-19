"""Classification metric evaluator."""

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from hyperagent.core.registries import evaluator_registry
from hyperagent.schemas import EvaluationReport


class ClassificationEvaluator:
    name = "classification"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationReport:
        y_true = np.asarray(y_true).astype(np.int64)
        y_pred = np.asarray(y_pred).astype(np.int64)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")
        labels = sorted(int(v) for v in np.unique(np.concatenate([y_true, y_pred])))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        total = int(matrix.sum())
        overall = float(np.trace(matrix) / total) if total else 0.0
        per_class = {}
        for idx, label in enumerate(labels):
            denom = int(matrix[idx].sum())
            per_class[str(label)] = float(matrix[idx, idx] / denom) if denom else 0.0
        average = float(np.mean(list(per_class.values()))) if per_class else 0.0
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels)) if total else 0.0
        return EvaluationReport(
            overall_accuracy=overall,
            average_accuracy=average,
            kappa=kappa,
            labels=labels,
            per_class_accuracy=per_class,
            confusion_matrix=matrix.astype(int).tolist(),
        )


evaluator_registry.register(ClassificationEvaluator.name, ClassificationEvaluator(), replace=True)

