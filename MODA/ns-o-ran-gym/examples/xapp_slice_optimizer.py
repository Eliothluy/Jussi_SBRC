#!/usr/bin/env python3
"""xApp Proof-of-Concept: O-RAN Slice Optimizer.

This script demonstrates how a trained offline policy would function as an
xApp in an O-RAN architecture. It receives network-state KPMs (Key Performance
Metrics) and returns optimized slice allocation weights.

CONCEPTUAL O-RAN FLOW (not real E2/A1 integration):
  1. Near-RT RIC collects KPMs from gNB via E2 interface
  2. xApp receives KPMs as "network_state" (num UEs, offered load per slice)
  3. xApp runs inference: neural policy -> optimal weights [eMBB, URLLC, MTC]
  4. xApp sends weights back to RIC -> gNB scheduler via A1 policy

Usage:
    cd ns-o-ran-gym
    python3 examples/xapp_slice_optimizer.py                 # demo with sac
    python3 examples/xapp_slice_optimizer.py --model ppo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.offline_models import (
    MODEL_TYPES,
    STATE_COLS,
    load_model,
    norm_params_from_checkpoint,
    predict_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (REPO_ROOT / "resultados_cenarios_finalizados_20260715"
                   / "offline_dataset_v2.parquet")
SLICE_NAMES = ["eMBB", "URLLC", "MTC"]


class SliceOptimizerXApp:
    """O-RAN xApp conceptual implementation for slice weight optimization.

    Trained offline from logged metaheuristic/heuristic data, this xApp
    provides instant (<1ms) slice weight recommendations given the current
    network state.

    In a real O-RAN deployment, this would run as a container in the Near-RT
    RIC, receiving KPMs via E2 and pushing policies via A1.
    """

    def __init__(self, model_path: Path, model_type: str = "sac",
                 dataset_path: Path | None = DEFAULT_DATASET):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.model_type = model_type
        self.model, ckpt = load_model(model_path, model_type)
        self.state_dim = ckpt["state_dim"]
        self.state_cols = ckpt.get("state_cols", STATE_COLS)
        self.action_table = ckpt.get("action_table")
        self.algo_description = ckpt.get("algo_description", "")
        # Normalization comes from the checkpoint; the dataset is only a
        # fallback for legacy checkpoints saved without norm_params.
        norm = norm_params_from_checkpoint(ckpt, dataset_path)
        self.norm_min = norm["min"]
        self.norm_range = norm["range"]

    def _normalize_state(self, raw_state: dict) -> np.ndarray:
        """Normalize raw KPM values to [0,1] using training-set statistics."""
        vals = []
        for col in STATE_COLS:
            v = raw_state.get(col, 0)
            v = (v - self.norm_min.get(col, 0)) / self.norm_range.get(col, 1)
            vals.append(max(0.0, min(1.0, v)))
        return np.array(vals, dtype=np.float32).reshape(1, -1)

    def predict_weights(self, network_state: dict) -> dict:
        """xApp entry point: receive KPMs, return optimal slice weights.

        Args:
            network_state: dict with keys like num_ues_total, offered_load_embb,
                           etc. Missing keys default to 0.

        Returns:
            dict with slice weights and metadata for the RIC policy.
        """
        x = self._normalize_state(network_state)
        weights = predict_weights(self.model, self.model_type, x, self.action_table)[0]

        # Ensure valid simplex
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()

        return {
            "slice_weights": {name: float(w) for name, w in zip(SLICE_NAMES, weights)},
            "policy_type": "static_slice_weights",
            "model": f"offline_{self.model_type}",
            "algo_description": self.algo_description,
            "inference_time_ms": "<1",
            "source": "trained_from_metaheuristic_and_heuristic_data",
        }


def demo(model_type: str, models_dir: Path, dataset: Path) -> None:
    """Demonstrate the xApp with all 4 scenarios."""
    print("=" * 75)
    print("  O-RAN xApp POC: Slice Optimizer (trained offline from logged results)")
    print("=" * 75)

    model_path = models_dir / f"{model_type}_offline.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run examples/train_offline_rl.py first.")
        return

    xapp = SliceOptimizerXApp(model_path, model_type=model_type, dataset_path=dataset)
    print(f"\nLoaded model: {model_path.name}")
    if xapp.algo_description:
        print(f"Algorithm: {xapp.algo_description}")
    print()

    # Define network states (simulating KPMs from E2 interface)
    test_states = [
        {
            "name": "Low Traffic (10 UEs, 57 Mbps offered)",
            "state": {
                "num_ues_total": 10, "num_ues_embb": 2, "num_ues_urllc": 2, "num_ues_mtc": 6,
                "offered_load_embb": 55.0, "offered_load_urllc": 1.0, "offered_load_mtc": 1.0,
                "pkt_size_embb": 1500, "pkt_size_urllc": 50, "pkt_size_mtc": 100,
                "is_low_traffic": 1, "is_normal": 0, "is_congestion": 0, "is_stressed": 0,
            },
        },
        {
            "name": "Normal (20 UEs, 73 Mbps offered)",
            "state": {
                "num_ues_total": 20, "num_ues_embb": 5, "num_ues_urllc": 5, "num_ues_mtc": 10,
                "offered_load_embb": 70.0, "offered_load_urllc": 1.0, "offered_load_mtc": 2.0,
                "pkt_size_embb": 1500, "pkt_size_urllc": 50, "pkt_size_mtc": 100,
                "is_low_traffic": 0, "is_normal": 1, "is_congestion": 0, "is_stressed": 0,
            },
        },
        {
            "name": "Congestion (60 UEs, 245 Mbps offered)",
            "state": {
                "num_ues_total": 60, "num_ues_embb": 15, "num_ues_urllc": 10, "num_ues_mtc": 35,
                "offered_load_embb": 180.0, "offered_load_urllc": 5.0, "offered_load_mtc": 60.0,
                "pkt_size_embb": 1500, "pkt_size_urllc": 100, "pkt_size_mtc": 500,
                "is_low_traffic": 0, "is_normal": 0, "is_congestion": 1, "is_stressed": 0,
            },
        },
        {
            "name": "Stressed (40 UEs, 128 Mbps offered)",
            "state": {
                "num_ues_total": 40, "num_ues_embb": 8, "num_ues_urllc": 12, "num_ues_mtc": 20,
                "offered_load_embb": 100.0, "offered_load_urllc": 3.0, "offered_load_mtc": 25.0,
                "pkt_size_embb": 1500, "pkt_size_urllc": 100, "pkt_size_mtc": 500,
                "is_low_traffic": 0, "is_normal": 0, "is_congestion": 0, "is_stressed": 1,
            },
        },
    ]

    for ts in test_states:
        print(f"\n--- {ts['name']} ---")
        result = xapp.predict_weights(ts["state"])
        w = result["slice_weights"]
        print("  Recommended weights:")
        for name in SLICE_NAMES:
            print(f"    {name + ':':<7}{w[name]:.3f}  ({w[name] * 100:.1f}%)")
        print(f"  Inference time: {result['inference_time_ms']} ms")

    print("\n" + "=" * 75)
    print("  xApp Architecture (O-RAN conceptual):")
    print("=" * 75)
    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │                    Near-RT RIC                              │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │  xApp: Slice Optimizer                               │  │
  │  │  ┌────────────┐    ┌──────────────┐    ┌──────────┐  │  │
  │  │  │ KPM Input  │───>│ Neural Policy│───>│ Weights  │  │  │
  │  │  │ (E2: UEs,  │    │ (offline-    │    │[eMBB,    │  │  │
  │  │  │  load/slice)│   │  trained     │    │ URLLC,   │  │  │
  │  │  │            │    │  policy)     │    │ MTC]     │  │  │
  │  │  └────────────┘    └──────────────┘    └────┬─────┘  │  │
  │  │                                            │A1      │  │
  │  └────────────────────────────────────────────┼────────┘  │
  └───────────────────────────────────────────────┼───────────┘
                                                   │
  ┌───────────────────────────────────────────────▼───────────┐
  │                    gNB Scheduler                           │
  │  slice_custom mode: floor(RBG_total × w_i) + remainder     │
  └────────────────────────────────────────────────────────────┘

  Training pipeline (offline):
  Meta-heuristics (GA/PSO/SA/Hybrid) + heuristic baselines
       ──>  Parquet dataset (context, weights, score)
       ──>  Offline policy training (Q-regression / BC / RWR)
       ──>  Neural policy (.pt)  ──>  xApp deployment
    """)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=list(MODEL_TYPES), default="sac")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Only used as normalization fallback for legacy checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(args.model, args.models_dir, args.dataset)
