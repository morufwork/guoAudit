"""Generic train/predict loop for any pair model with forward(h_a, h_b) -> logits.

Shared by src/models/siamese.py, attention.py and interaction.py so the training procedure -- Adam, early stopping on an internal
validation split carved from training data only (test set never touched for
stopping) -- is identical across every architecture compared in
the ablation. Only the model's forward pass differs between them.
"""
from typing import Callable

import numpy as np
import torch
from torch import nn


def train_pair_model(
    model_fn: Callable[[], nn.Module],
    h_a_train: np.ndarray, h_b_train: np.ndarray, y_train: np.ndarray,
    seed: int,
    val_fraction: float = 0.15, max_epochs: int = 100, patience: int = 10,
    batch_size: int = 128, lr: float = 1e-3,
) -> nn.Module:
    """`model_fn` builds a fresh, uninitialized model -- construction must
    happen AFTER the seed is set, or weight initialization silently depends
    on whatever the ambient torch RNG state happens to be (i.e. on prior,
    unrelated calls elsewhere in the process), breaking reproducibility."""
    torch.manual_seed(seed)
    model = model_fn()
    rng = np.random.default_rng(seed)

    n = len(y_train)
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def to_tensors(idx):
        return (
            torch.tensor(h_a_train[idx], dtype=torch.float32),
            torch.tensor(h_b_train[idx], dtype=torch.float32),
            torch.tensor(y_train[idx], dtype=torch.float32),
        )

    a_tr, b_tr, y_tr = to_tensors(train_idx)
    a_val, b_val, y_val = to_tensors(val_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    n_train = len(y_tr)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(a_tr[batch_idx], b_tr[batch_idx])
            loss = loss_fn(logits, y_tr[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(a_val, b_val), y_val).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_proba(model: nn.Module, h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(h_a, dtype=torch.float32), torch.tensor(h_b, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()
