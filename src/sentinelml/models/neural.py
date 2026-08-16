"""Small PyTorch MLP for tabular fraud scores, with sklearn MLP fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sentinelml.models.base import ensure_2d_proba

HAS_TORCH = False
torch = None
nn = None
try:
    import torch as _torch
    from torch import nn as _nn

    torch = _torch
    nn = _nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


if HAS_TORCH and nn is not None:

    class FraudMLP(nn.Module):
        def __init__(self, n_in: int, hidden: list[int] | None = None) -> None:
            super().__init__()
            hidden = hidden or [32, 16]
            layers: list[nn.Module] = []
            prev = n_in
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x).squeeze(-1)

else:  # still export a name so imports do not fail when torch is absent

    class FraudMLP:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is not installed")


class TorchMLPModel:
    name = "mlp_torch"
    version = "0.1.0"

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        epochs: int = 12,
        batch_size: int = 256,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.hidden_dims = hidden_dims or [32, 16]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model: Any = None
        self.device = device if device in {"cpu", "cuda"} else "cpu"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if not HAS_TORCH or torch is None or nn is None:
            raise RuntimeError("PyTorch is not installed")
        xs = self.scaler.fit_transform(x).astype(np.float32)
        ys = y.astype(np.float32)
        self.model = FraudMLP(n_in=xs.shape[1], hidden=self.hidden_dims).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        pos = max(float(ys.sum()), 1.0)
        neg = max(float(len(ys) - pos), 1.0)
        pos_weight = torch.tensor([neg / pos], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        tensor_x = torch.from_numpy(xs).to(self.device)
        tensor_y = torch.from_numpy(ys).to(self.device)
        n = len(xs)
        self.model.train()
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        for _ in range(self.epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch_x = tensor_x[idx]
                batch_y = tensor_y[idx]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = self.model(batch_x)
                    loss = loss_fn(logits, batch_y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not HAS_TORCH or torch is None or self.model is None:
            raise RuntimeError("Torch MLP is not fitted")
        xs = self.scaler.transform(x).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(xs).to(self.device))
            proba = torch.sigmoid(logits).cpu().numpy()
        return np.clip(np.asarray(proba, dtype=np.float64), 0.0, 1.0)

    def get_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "framework": "torch",
            "device": self.device,
            "hidden_dims": self.hidden_dims,
        }


class SklearnMLPModel:
    name = "mlp_sklearn"
    version = "0.1.0"

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (32, 16), max_iter: int = 80) -> None:
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        xs = self.scaler.fit_transform(x)
        self.model.fit(xs, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xs = self.scaler.transform(x)
        return ensure_2d_proba(self.model.predict_proba(xs))

    def get_metadata(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "framework": "sklearn"}
