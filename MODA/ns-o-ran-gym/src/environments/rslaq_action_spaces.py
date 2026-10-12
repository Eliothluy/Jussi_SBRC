"""
RSLAQ Action Spaces.

Handles conversion from agent actions to PRB percentages for ns-3,
supporting both continuous (SAC) and discrete (DDQN) modes.

P_STA decomposition: p_final = static_fraction * weights + (1 - static_fraction) * p_opt

IMPORTANT: When using this module with ns-3 DRL training, set apply_p_sta=False to avoid
double application. P_STA is applied on the Python side when the agent produces actions,
and ns-3 should receive the final PRB percentages directly without further modification.
"""

from typing import List, Tuple
import numpy as np

# Default static weights from the RSLAQ paper
DEFAULT_WEIGHTS = np.array([0.3333, 0.4000, 0.2667], dtype=np.float32)


def _prepare_p_sta_params(
    weights: np.ndarray | None,
    static_fraction: float,
) -> tuple[np.ndarray, float]:
    """Validate and normalize P_STA weights and static-fraction inputs."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    weights = np.asarray(weights, dtype=np.float64).flatten()
    if weights.shape[0] != 3:
        raise ValueError(f"weights must have shape (3,), got {weights.shape}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    total = weights.sum()
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    weights = weights / total

    static_fraction = float(static_fraction)
    if not np.isfinite(static_fraction):
        raise ValueError("static_fraction must be finite")
    if static_fraction < 0.0 or static_fraction > 1.0:
        raise ValueError("static_fraction must be in [0, 1]")
    return weights, static_fraction


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def build_discrete_action_table(
    step: float = 0.1,
    include_scheduler: bool = False,
) -> List[Tuple]:
    """
    Build the discrete action table for DDQN.

    Each action is a tuple (p0, p1, p2, [sch]) where p0+p1+p2 ≈ 1.0
    and p_i are multiples of ``step``.

    With step=0.1 and 3 slices (resource-only), there are 66 actions.
    If include_scheduler=True, actions also include scheduler_id (0=RR, 1=PF, 2=BCQI),
    resulting in 198 actions.

    Args:
        step: Granularity of resource allocation (default 0.1).
        include_scheduler: Whether to include scheduler selection (default False).

    Returns:
        List of action tuples.
    """
    values = [round(i * step, 1) for i in range(int(1.0 / step) + 1)]
    combos: List[Tuple] = []
    for p0 in values:
        for p1 in values:
            p2 = round(1.0 - p0 - p1, 1)
            if p2 < -1e-6 or p2 > 1.0 + 1e-6:
                continue
            if abs(p0 + p1 + p2 - 1.0) > 1e-6:
                continue
            if include_scheduler:
                for sch in (0, 1, 2):  # RR, PF, BCQI
                    combos.append((float(p0), float(p1), float(p2), int(sch)))
            else:
                combos.append((float(p0), float(p1), float(p2)))
    return combos


def continuous_action_to_prb(
    raw_action: np.ndarray,
    weights: np.ndarray = None,
    static_fraction: float = 0.5,
    apply_p_sta: bool = True,
) -> np.ndarray:
    """
    Convert a continuous raw action (e.g. from SAC) to PRB percentages.

    Process:
        1. Softmax on raw_action
        2. p_opt = softmax(raw_action)
        3. If apply_p_sta=True: p_final = static_fraction * weights + (1 - static_fraction) * p_opt
           If apply_p_sta=False: p_final = p_opt (raw agent output)
        4. Normalize to sum 1.0
        5. Convert to percentages (*100)

    Args:
        raw_action: Array of shape (3,) in [-1, 1] or any real range.
        weights: Static weights (default [0.3333, 0.4000, 0.2667]).
        static_fraction: Fraction allocated to static weights (default 0.5).
        apply_p_sta: If True, apply P_STA decomposition. If False, return raw softmax output.
                    Set to False when ns-3 handles P_STA decomposition. Default: True for backward compatibility.

    Returns:
        Array of shape (3,) with PRB percentages summing to ~100.0.
    """
    weights, static_fraction = _prepare_p_sta_params(weights, static_fraction)
    raw_action = np.asarray(raw_action, dtype=np.float64).flatten()

    if raw_action.shape[0] != 3:
        raise ValueError(f"raw_action must have shape (3,), got {raw_action.shape}")
    if not np.all(np.isfinite(raw_action)):
        raise ValueError("raw_action must contain only finite values")

    p_opt = _softmax(raw_action)
    if apply_p_sta:
        p_final = static_fraction * weights + (1.0 - static_fraction) * p_opt
    else:
        p_final = p_opt
    p_final /= p_final.sum()
    prb_pct = p_final * 100.0
    return prb_pct.astype(np.float64)


def discrete_action_to_prb(
    action_idx: int,
    action_table: List[Tuple],
    weights: np.ndarray = None,
    static_fraction: float = 0.5,
    apply_p_sta: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Convert a discrete action index (DDQN) to PRB percentages.

    Args:
        action_idx: Index into the action table.
        action_table: List of action tuples from build_discrete_action_table.
        weights: Static weights (default [0.3333, 0.4000, 0.2667]).
        static_fraction: Fraction allocated to static weights (default 0.5).
        apply_p_sta: If True, apply P_STA decomposition. If False, return raw action table values.
                    Set to False when ns-3 handles P_STA decomposition. Default: True for backward compatibility.

    Returns:
        Tuple (prb_percentages, scheduler_id).
        If the action table does not include scheduler, scheduler_id is -1.
    """
    weights, static_fraction = _prepare_p_sta_params(weights, static_fraction)

    if action_idx < 0 or action_idx >= len(action_table):
        raise ValueError(f"action_idx {action_idx} out of range [0, {len(action_table)})")

    entry = action_table[action_idx]
    if len(entry) == 3:
        p0, p1, p2 = entry
        sch_id = -1
    elif len(entry) == 4:
        p0, p1, p2, sch_id = entry
    else:
        raise ValueError(f"Unexpected action table entry format: {entry}")

    p_opt = np.array([p0, p1, p2], dtype=np.float64)
    if not np.all(np.isfinite(p_opt)) or np.any(p_opt < 0.0):
        raise ValueError(f"Action table entry must contain non-negative finite values: {entry}")
    if p_opt.sum() <= 0.0:
        raise ValueError(f"Action table entry must sum to a positive value: {entry}")
    # Ensure it sums to 1.0 (guard against float rounding)
    p_opt /= p_opt.sum()

    if apply_p_sta:
        p_final = static_fraction * weights + (1.0 - static_fraction) * p_opt
    else:
        p_final = p_opt
    p_final /= p_final.sum()
    prb_pct = p_final * 100.0
    return prb_pct.astype(np.float64), int(sch_id)


def scheduler_id_to_name(scheduler_id: int) -> str:
    """Map scheduler id to human-readable name and cost."""
    mapping = {
        0: ("RR", 1.0),   # Round Robin
        1: ("PF", 2.0),   # Proportional Fair
        2: ("BCQI", 2.0)  # Best CQI
    }
    name, cost = mapping.get(scheduler_id, ("UNKNOWN", 1.0))
    return f"{name}(cost={cost})"
