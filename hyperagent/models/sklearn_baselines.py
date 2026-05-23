"""Additional sklearn baselines for reproducible benchmark matrices."""

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from hyperagent.core.registries import model_registry


class RandomForestBaseline:
    def __init__(self, params: Dict[str, Any], seed: int) -> None:
        self.model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth"),
            class_weight=params.get("class_weight"),
            random_state=seed,
            n_jobs=int(params.get("n_jobs", 1)),
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class KNNBaseline:
    def __init__(self, params: Dict[str, Any]) -> None:
        self.model = KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=str(params.get("weights", "distance")),
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


def build_random_forest(params: Dict[str, Any], seed: int) -> RandomForestBaseline:
    return RandomForestBaseline(params, seed)


def build_knn(params: Dict[str, Any], seed: int) -> KNNBaseline:
    del seed
    return KNNBaseline(params)


model_registry.register("random_forest", build_random_forest, replace=True)
model_registry.register("knn", build_knn, replace=True)
