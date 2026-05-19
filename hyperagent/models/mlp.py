"""Lightweight PyTorch MLP baseline."""

from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyperagent.core.registries import model_registry


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier:
    def __init__(self, params: Dict[str, Any], seed: int) -> None:
        self.params = params
        self.seed = seed
        self.labels: List[int] = []
        self.model: _MLP = None  # type: ignore[assignment]

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        self.labels = sorted(int(v) for v in np.unique(y_train))
        label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        y_index = np.asarray([label_to_index[int(v)] for v in y_train], dtype=np.int64)
        hidden_dim = int(self.params.get("hidden_dim", 64))
        epochs = int(self.params.get("epochs", 30))
        lr = float(self.params.get("lr", 0.001))
        batch_size = int(self.params.get("batch_size", 128))

        self.model = _MLP(x_train.shape[1], hidden_dim, len(self.labels))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_index, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for _ in range(epochs):
            for features, target in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(features), target)
                loss.backward()
                optimizer.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLPClassifier must be fitted before predict")
        self.model.eval()
        outputs = []
        batch_size = int(self.params.get("batch_size", 128))
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
                pred = torch.argmax(self.model(batch), dim=1).cpu().numpy()
                outputs.extend(self.labels[int(idx)] for idx in pred)
        return np.asarray(outputs, dtype=np.int64)


def build_mlp(params: Dict[str, Any], seed: int) -> MLPClassifier:
    return MLPClassifier(params, seed)


model_registry.register("mlp", build_mlp, replace=True)

