"""SVM baseline model."""

from typing import Any, Dict

import numpy as np
from sklearn.svm import SVC

from hyperagent.core.registries import model_registry


class SVMClassifier:
    def __init__(self, params: Dict[str, Any]) -> None:
        self.model = SVC(
            kernel=params.get("kernel", "rbf"),
            C=float(params.get("C", 10.0)),
            gamma=params.get("gamma", "scale"),
            class_weight=params.get("class_weight"),
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


def build_svm(params: Dict[str, Any], seed: int) -> SVMClassifier:
    del seed
    return SVMClassifier(params)


model_registry.register("svm", build_svm, replace=True)

