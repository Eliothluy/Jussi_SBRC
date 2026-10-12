"""Shared model definitions for the offline slice-weight policies (xApp line).

Single source of truth for the three offline policies trained from logged
metaheuristic/heuristic results, imported by examples/train_offline_rl.py,
examples/eval_offline_rl.py and examples/xapp_slice_optimizer.py so that the
architectures, normalization and checkpoint format can never drift apart.

Honest naming: the logged data has no next-state, so none of these models is
trained with TD-learning. They are contextual-bandit / supervised policies:
  - "ddqn": Q-regression over 66 discretized simplex actions
  - "sac":  behavioral cloning of top-quantile candidates
  - "ppo":  reward-weighted regression (RWR)
The ddqn/sac/ppo labels are kept only as file/CLI identifiers.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")

# State features (input to all models)
STATE_COLS = [
    "num_ues_total", "num_ues_embb", "num_ues_urllc", "num_ues_mtc",
    "offered_load_embb", "offered_load_urllc", "offered_load_mtc",
    "pkt_size_embb", "pkt_size_urllc", "pkt_size_mtc",
    "is_low_traffic", "is_normal", "is_congestion", "is_stressed",
]
ACTION_COLS = ["w_embb", "w_urllc", "w_mtc"]
SCENARIOS = ["low_traffic", "normal", "congestion", "stressed"]
SLICE_NAMES = ["eMBB", "URLLC", "MTC"]

MODEL_TYPES = ("ddqn", "sac", "ppo")

ALGO_DESCRIPTIONS = {
    "ddqn": ("DDQN-offline: contextual-bandit Q-regression over 66 simplex "
             "bins (no TD-learning, no MDP)"),
    "sac": ("SAC-offline: behavioral cloning of top-20% candidates per "
            "scenario (supervised, no critic)"),
    "ppo": ("PPO-offline: reward-weighted regression (RWR, tau=0.1; no "
            "policy-gradient rollouts)"),
}


# ---------------------------------------------------------------------------
# Discretized action table
# ---------------------------------------------------------------------------
def build_simplex_actions(step: float = 0.1) -> np.ndarray:
    """Generate all weight vectors on the 3-simplex with given step."""
    actions = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            actions.append([i * step, j * step, k * step])
    return np.array(actions, dtype=np.float32)


def nearest_action_index(weights: np.ndarray, action_table: np.ndarray) -> int:
    """Find the index of the nearest discretized action."""
    dists = np.sum((action_table - weights) ** 2, axis=1)
    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Architectures (must stay identical to the originally committed checkpoints)
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)  # ensures simplex

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class PolicyNet(nn.Module):
    """Gaussian policy trunk with softmax mean projection to the simplex."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, 3)
        self.log_std = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std.expand_as(mean).clamp(-3, 0)
        return mean, log_std

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(x)
        return F.softmax(mean, dim=-1)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def compute_norm_params(df: pd.DataFrame) -> dict:
    """Min-max normalization params over STATE_COLS (zero range -> 1)."""
    state_df = df[STATE_COLS]
    state_min = state_df.min()
    state_range = (state_df.max() - state_min).replace(0, 1)
    return {"min": state_min.to_dict(), "range": state_range.to_dict()}


def normalize_state(raw: np.ndarray, norm_params: dict) -> np.ndarray:
    """Normalize a (batch, len(STATE_COLS)) array with stored params."""
    mins = np.array([norm_params["min"][c] for c in STATE_COLS], dtype=np.float32)
    ranges = np.array([norm_params["range"][c] for c in STATE_COLS], dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    return (raw - mins) / ranges


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    model_type: str,
    state_dim: int,
    norm_params: dict,
    loss_history: list[float],
    train_seed: int,
    dataset_path: str = "",
    dataset_version: str = "",
    action_table: np.ndarray | None = None,
    extras: dict | None = None,
) -> None:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    payload: dict = {
        "state_dict": model.state_dict(),
        "state_dim": state_dim,
        "state_cols": STATE_COLS,
        "loss_history": loss_history,
        "model_type": model_type,
        "algo_description": ALGO_DESCRIPTIONS[model_type],
        "norm_params": norm_params,
        "dataset_path": dataset_path,
        "dataset_version": dataset_version,
        "train_seed": train_seed,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if model_type == "ddqn":
        if action_table is None:
            raise ValueError("ddqn checkpoints require an action_table")
        payload["n_actions"] = len(action_table)
        payload["action_table"] = action_table
    if extras:
        payload.update(extras)
    torch.save(payload, path)


def load_model(path: Path, model_type: str) -> tuple[nn.Module, dict]:
    """Load a checkpoint; returns (model, checkpoint_dict).

    Backward compatible with pre-refactor checkpoints (which lack norm_params
    and algo_description); callers must handle a missing "norm_params" key,
    typically via norm_params_from_checkpoint().
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dim = ckpt["state_dim"]
    if model_type == "ddqn":
        model: nn.Module = QNet(state_dim, ckpt["n_actions"])
    elif model_type == "sac":
        model = ActorNet(state_dim)
    else:
        model = PolicyNet(state_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def norm_params_from_checkpoint(ckpt: dict, dataset_path: Path | None = None) -> dict:
    """Return checkpoint norm_params; fall back to recomputing from a dataset.

    The fallback exists only for checkpoints created before norm_params were
    stored; it is correct only if the dataset is the same one used to train.
    """
    params = ckpt.get("norm_params")
    if params:
        return params
    if dataset_path is None:
        raise ValueError(
            "Checkpoint lacks norm_params and no dataset was given for fallback")
    warnings.warn(
        f"Checkpoint lacks norm_params; recomputing from {dataset_path}. "
        "Results are only correct if this is the training dataset.",
        stacklevel=2,
    )
    df = pd.read_parquet(dataset_path)
    return compute_norm_params(df)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_weights(
    model: nn.Module,
    model_type: str,
    state: np.ndarray,
    action_table: np.ndarray | None = None,
) -> np.ndarray:
    """Predict simplex weights for a batch of normalized states."""
    model.eval()
    x = torch.tensor(np.asarray(state, dtype=np.float32), device=DEVICE)
    with torch.no_grad():
        if model_type == "ddqn":
            if action_table is None:
                raise ValueError("ddqn prediction requires an action_table")
            idx = model(x).argmax(dim=1).cpu().numpy()
            return np.asarray(action_table)[idx]
        if model_type == "sac":
            return model(x).cpu().numpy()
        if model_type == "ppo":
            return model.predict(x).cpu().numpy()
    raise ValueError(f"Unknown model type: {model_type}")
