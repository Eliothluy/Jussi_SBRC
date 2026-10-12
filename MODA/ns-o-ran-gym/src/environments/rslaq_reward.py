"""
RSLAQ SLA-aware Reward Function.

Computes the reward based on per-slice KPIs and scenario-specific SLA targets,
strictly following the equations published in the RSLAQ paper (IEEE TMC 2026):

- Equations 16: h_1 = (1/|Λ1|) * Σ thr(UE) normalized by max achievable rate (eMBB)
- Equation 17: h_2 = exp(-max_UE bfs(UE) / norm) (URLLC)
- Equation 18: h_3 = (1/|Λ3|) * Σ thr(UE) normalized by max achievable rate (MTC)
- Equation 8:  ropt = alpha*h_1 + beta*h_2 + gamma*h_3 + 1/scheduler_cost
- Equation 12: Terminal conditions (soft and outage SLA violations)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .rslaq_slice_ids import get_num_slices, slice_name

# Default importance weights (omega_j in the paper, Eq. 8)
# Priorities [2, 1, 3] for [eMBB, URLLC, MTC] → weights [0.33, 0.40, 0.27]
ALPHA = 0.3333  # omega_0: eMBB
BETA = 0.4000   # omega_1: URLLC
GAMMA = 0.2667  # omega_2: MTC

# Default max achievable rate per UE for throughput normalization (Mbps)
# Paper: "all elements must be normalized (e.g., we use maximum
# achievable rates for the throughput normalization)."
DEFAULT_MAX_ACHIEVABLE_THR_MBPS = 150.0

# SLA targets per scenario (matching Table V in the paper)
SLA_BY_SCENARIO: Dict[str, Dict[str, Any]] = {
    "low_traffic": {
        "embb_min_throughput_mbps": 10.0,
        "embb_soft_max_throughput_mbps": 15.0,
        "urllc_outage_bfs_bytes": 10000.0,
        "mtc_max_tdp": 1000.0,
        "mtc_target_throughput": 10.0,
        # MTC is No-Policy in all paper scenarios (Table V)
        "mtc_no_policy": True,
    },
    "normal": {
        "embb_min_throughput_mbps": 10.0,
        "embb_soft_max_throughput_mbps": 15.0,
        "urllc_outage_bfs_bytes": 10000.0,
        "mtc_max_tdp": 1000.0,
        "mtc_target_throughput": 10.0,
        "mtc_no_policy": True,
    },
    "congestion": {
        "embb_min_throughput_mbps": 10.0,
        "embb_soft_max_throughput_mbps": 15.0,
        "urllc_outage_bfs_bytes": 10000.0,
        "mtc_max_tdp": 1000.0,
        "mtc_target_throughput": 20.0,
        "mtc_no_policy": True,
    },
    "stressed": {
        "embb_min_throughput_mbps": 20.0,
        "embb_soft_max_throughput_mbps": 25.0,
        "urllc_outage_bfs_bytes": 10000.0,
        "mtc_max_tdp": 1000.0,
        "mtc_target_throughput": 20.0,
        "mtc_no_policy": True,
    },
    "insufficient_resources": {
        "embb_min_throughput_mbps": 20.0,
        "embb_soft_max_throughput_mbps": 25.0,
        "urllc_outage_bfs_bytes": 10000.0,
        "mtc_max_tdp": 1000.0,
        "mtc_target_throughput": 20.0,
        "mtc_no_policy": True,
    },
}


@dataclass
class RewardResult:
    """Container returned by the RSLAQ reward computation."""

    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    outage_flags: Dict[int, bool] = field(default_factory=dict)
    soft_flags: Dict[int, bool] = field(default_factory=dict)
    optimization_terms: Dict[str, float] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)


def get_scheduler_cost(scheduler_id: int) -> float:
    """
    Returns the cost of the scheduler (Eq. 9 in the paper).

    - RR (Round Robin): cost 1.0
    - PF (Proportional Fair): cost 2.0
    - BCQI (Best CQI): cost 2.0
    """
    mapping = {0: 1.0, 1: 2.0, 2: 2.0}
    return mapping.get(scheduler_id, 1.0)


def _evaluate_outage_conditions(
    metrics: Dict[int, Dict[str, Any]],
    sla: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[int, bool]:
    """
    Evaluate instantaneous outage conditions for each slice (paper Eq. 13-14).

    eMBB: P(thr per slice < k_11) where k_11 = min throughput
    URLLC: P(bfs per UE > k_12) where k_12 = max buffer occupancy
    MTC: No outage condition (no-policy slice in paper Table V)

    Returns a dict {slice_id: bool} where True means the slice meets
    the instantaneous outage condition in this step.
    """
    flags: Dict[int, bool] = {0: False, 1: False, 2: False}

    embb = metrics.get(0, {})
    urllc = metrics.get(1, {})

    # eMBB outage (Eq. 13): throughput below minimum AND demand exists.
    # The paper-faithful default uses the absolute SLA threshold. Predictive
    # experiments can opt into a demand-aware cap to avoid penalizing a slice
    # for not exceeding the offered traffic in low-load periods.
    embb_thr = float(embb.get("throughputMbps_sum", 0.0))
    embb_tx = float(embb.get("dTxBytes_sum", 0.0))
    min_tx = float(config.get("min_tx_bytes_for_outage", 1.0))
    embb_threshold = float(sla["embb_min_throughput_mbps"])
    if config.get("demand_aware_embb_outage", False):
        period_ms = float(config.get("period_ms", config.get("periodMs", 10.0)))
        if period_ms > 0:
            offered_mbps = embb_tx * 8.0 / (period_ms * 1000.0)
            embb_threshold = min(embb_threshold, offered_mbps)
    tolerance = float(config.get("embb_outage_tolerance_mbps", 1e-6))
    if embb_thr + tolerance < embb_threshold and embb_tx >= min_tx:
        flags[0] = True

    # URLLC outage (Eq. 14): buffer exceeds threshold
    urllc_max_bfs = float(urllc.get("bufferBytes_max", 0.0))
    if urllc_max_bfs > sla["urllc_outage_bfs_bytes"]:
        flags[1] = True

    # MTC is No-Policy in paper Table V → no outage condition
    # (slice 2 is always False)

    return flags


def _evaluate_soft_conditions(
    metrics: Dict[int, Dict[str, Any]],
    sla: Dict[str, Any],
) -> Dict[int, bool]:
    """
    Evaluate instantaneous soft SLA conditions (paper Eq. 15).

    eMBB: P(max thr per slice > k_21) where k_21 = soft max throughput
    URLLC: No soft condition defined in paper for URLLC
    MTC: No-Policy → no soft condition
    """
    flags: Dict[int, bool] = {0: False, 1: False, 2: False}

    embb = metrics.get(0, {})

    # eMBB soft (Eq. 15): throughput exceeds soft max → wasteful overallocation
    embb_thr = float(embb.get("throughputMbps_sum", 0.0))
    soft_max = sla.get("embb_soft_max_throughput_mbps")
    if soft_max is not None and embb_thr > soft_max:
        flags[0] = True

    return flags


def _normalize_reward_mode(mode: Any) -> str:
    """Return the canonical reward mode name from user-facing aliases."""
    normalized = str(mode or "paper").strip().lower().replace("-", "_")
    if normalized in ("paper", "rslaq", "rslaq_paper"):
        return "paper"
    if normalized in (
        "resource_efficient",
        "efficient_resource",
        "efficiente_resource",
        "efficient",
        "slice_efficient",
    ):
        return "resource_efficient"
    raise ValueError(
        f"Unknown reward_mode={mode!r}. Use 'paper' or 'resource_efficient'."
    )


def _normalize_vector(values: List[float], fallback: List[float]) -> List[float]:
    """Normalize non-negative values, falling back to defaults when empty."""
    clean = [max(float(v), 0.0) for v in values]
    total = sum(clean)
    if total <= 0.0:
        clean = [max(float(v), 0.0) for v in fallback]
        total = sum(clean)
    if total <= 0.0:
        return [1.0 / get_num_slices()] * get_num_slices()
    return [v / total for v in clean]


def _extract_static_weights(action_info: Optional[Dict[str, Any]]) -> List[float]:
    """Extract normalized P_STA static weights from action metadata."""
    if action_info and "p_sta_weights" in action_info:
        try:
            weights = [float(v) for v in action_info["p_sta_weights"][:get_num_slices()]]
            return _normalize_vector(weights, [ALPHA, BETA, GAMMA])
        except (TypeError, ValueError):
            pass
    return _normalize_vector([ALPHA, BETA, GAMMA], [ALPHA, BETA, GAMMA])


def _extract_allocation_share(
    metrics: Dict[int, Dict[str, Any]],
    action_info: Optional[Dict[str, Any]],
) -> tuple[List[float], List[float]]:
    """Return raw PRB percentages and normalized allocation shares."""
    allocation_pct: List[float] = []
    if action_info and "prb_pct" in action_info:
        try:
            allocation_pct = [
                max(float(v), 0.0)
                for v in action_info["prb_pct"][:get_num_slices()]
            ]
        except (TypeError, ValueError):
            allocation_pct = []

    if len(allocation_pct) != get_num_slices() or sum(allocation_pct) <= 0.0:
        allocation_pct = [
            max(float(metrics.get(sid, {}).get("resourceSharePct_mean", 0.0)), 0.0)
            for sid in range(get_num_slices())
        ]

    fallback = [100.0 / get_num_slices()] * get_num_slices()
    allocation_share = _normalize_vector(allocation_pct, fallback)
    return allocation_pct, allocation_share


def _compute_resource_efficiency_terms(
    metrics: Dict[int, Dict[str, Any]],
    sla: Dict[str, Any],
    action_info: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute resource-efficiency shaping terms.

    This is deliberately a shaping layer over the paper reward: it measures
    whether PRBs follow active per-slice demand and penalizes waste, but it
    does not override the RSLAQ terminal SLA logic.
    """
    num_slices = get_num_slices()
    static_weights = _extract_static_weights(action_info)
    allocation_pct, allocation_share = _extract_allocation_share(metrics, action_info)

    loss_packet_byte_equivalent = float(
        config.get("loss_packet_byte_equivalent", 1500.0)
    )
    dynamic_need_weight = float(config.get("resource_dynamic_need_weight", 0.75))
    dynamic_need_weight = min(max(dynamic_need_weight, 0.0), 1.0)

    raw_need: List[float] = []
    served_fraction: List[float] = []
    for sid in range(num_slices):
        slice_metrics = metrics.get(sid, {})
        tx_bytes = max(float(slice_metrics.get("dTxBytes_sum", 0.0)), 0.0)
        rx_bytes = max(float(slice_metrics.get("dRxBytes_sum", 0.0)), 0.0)
        buffer_bytes = max(float(slice_metrics.get("bufferBytes_max", 0.0)), 0.0)
        lost_packets = max(float(slice_metrics.get("dLostPackets_sum", 0.0)), 0.0)
        throughput = max(float(slice_metrics.get("throughputMbps_sum", 0.0)), 0.0)

        need = tx_bytes + buffer_bytes + lost_packets * loss_packet_byte_equivalent

        if sid == 0:
            target = float(sla["embb_min_throughput_mbps"])
            if target > 0.0 and tx_bytes > 0.0 and throughput < target:
                deficit = min((target - throughput) / target, 1.0)
                need *= 1.0 + deficit
        elif sid == 1:
            threshold = float(sla["urllc_outage_bfs_bytes"])
            if threshold > 0.0:
                buffer_pressure = min(buffer_bytes / threshold, 2.0)
                need += buffer_pressure * threshold
        elif sid == 2:
            target = float(sla.get("mtc_target_throughput", 1.0))
            if target > 0.0 and tx_bytes > 0.0 and throughput < target:
                deficit = min((target - throughput) / target, 1.0)
                need *= 1.0 + 0.5 * deficit

        raw_need.append(max(need, 0.0))

        if tx_bytes > 1.0:
            served_fraction.append(min(rx_bytes / tx_bytes, 1.0))
        else:
            served_fraction.append(1.0 if need <= 1e-9 else 0.0)

    demand_share = _normalize_vector(raw_need, static_weights)
    need_share = [
        (1.0 - dynamic_need_weight) * static_weights[sid]
        + dynamic_need_weight * demand_share[sid]
        for sid in range(num_slices)
    ]
    need_share = _normalize_vector(need_share, static_weights)

    allocation_distance = 0.5 * sum(
        abs(allocation_share[sid] - need_share[sid]) for sid in range(num_slices)
    )
    need_allocation_match = max(0.0, 1.0 - allocation_distance)

    deadband = float(config.get("resource_waste_deadband", 0.03))
    deadband = max(deadband, 0.0)
    over_allocation = sum(
        max(allocation_share[sid] - need_share[sid] - deadband, 0.0)
        for sid in range(num_slices)
    )
    under_allocation = sum(
        max(need_share[sid] - allocation_share[sid] - deadband, 0.0)
        for sid in range(num_slices)
    )

    served_score = sum(
        need_share[sid] * served_fraction[sid] for sid in range(num_slices)
    )
    resource_efficiency = served_score * need_allocation_match

    embb_soft_guard_penalty = 0.0
    embb_soft_max = sla.get("embb_soft_max_throughput_mbps")
    if embb_soft_max is not None:
        soft_max = float(embb_soft_max)
        guard_ratio = float(config.get("embb_soft_guard_ratio", 0.80))
        guard_ratio = min(max(guard_ratio, 0.0), 1.0)
        guard_start = soft_max * guard_ratio
        embb_metrics = metrics.get(0, {})
        embb_tx = max(float(embb_metrics.get("dTxBytes_sum", 0.0)), 0.0)
        embb_thr = max(float(embb_metrics.get("throughputMbps_sum", 0.0)), 0.0)
        if soft_max > guard_start and embb_tx > 1.0 and embb_thr > guard_start:
            pressure = min((embb_thr - guard_start) / (soft_max - guard_start), 1.0)
            embb_soft_guard_penalty = pressure * allocation_share[0]

    action_smoothness_penalty = 0.0
    if action_info and "previous_prb_pct" in action_info:
        try:
            previous_pct = [
                max(float(v), 0.0)
                for v in action_info["previous_prb_pct"][:num_slices]
            ]
            previous_share = _normalize_vector(previous_pct, allocation_share)
            action_smoothness_penalty = 0.5 * sum(
                abs(allocation_share[sid] - previous_share[sid])
                for sid in range(num_slices)
            )
        except (TypeError, ValueError):
            action_smoothness_penalty = 0.0

    terms = {
        "resource_efficiency": resource_efficiency,
        "need_allocation_match": need_allocation_match,
        "over_allocation": over_allocation,
        "under_allocation": under_allocation,
        "embb_soft_guard_penalty": embb_soft_guard_penalty,
        "action_smoothness_penalty": action_smoothness_penalty,
    }
    debug = {
        "allocation_pct": allocation_pct,
        "allocation_share": allocation_share,
        "static_weights": static_weights,
        "raw_need": raw_need,
        "demand_share": demand_share,
        "need_share": need_share,
        "served_fraction": served_fraction,
    }
    return terms, debug


def compute_rslaq_reward(
    metrics: Dict[int, Dict[str, Any]],
    scenario: str,
    action_info: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    step_count: int = 0,
    kpi_history: Optional[List[Dict[int, Dict[str, Any]]]] = None,
) -> RewardResult:
    """
    Compute the SLA-aware reward for RSLAQ following the paper.

    Equations implemented:
        - Eq. 16: h_1 = per-UE avg throughput / max achievable rate (eMBB)
        - Eq. 17: h_2 = exp(-max_bfs / bfs_norm) (URLLC)
        - Eq. 18: h_3 = per-UE avg throughput / max achievable rate (MTC)
        - Eq. 8:  opt_reward = (alpha*h_1 + beta*h_2 + gamma*h_3) + (1/scheduler_cost)
        - Eq. 12: Terminal conditions (soft and outage)

    Paper says: "all elements must be normalized (e.g., we use maximum
    achievable rates for the throughput normalization)."
    "h_2 = exp(-max_UE bfs(UE))" with bfs normalized.

    Args:
        step_count: Current step within the episode. Terminal conditions are
            suppressed during the warmup period.
    """
    if config is None:
        config = {}

    reward_mode = _normalize_reward_mode(config.get("reward_mode", "paper"))
    sla = SLA_BY_SCENARIO.get(scenario, SLA_BY_SCENARIO["normal"])
    warmup_steps = int(config.get("warmup_steps", 5))
    consecutive_outage_steps = int(config.get("consecutive_outage_steps", 5))
    terminate_on_sla_violation = bool(config.get("terminate_on_sla_violation", True))

    alpha = config.get("alpha", ALPHA)
    beta = config.get("beta", BETA)
    gamma_val = config.get("gamma", GAMMA)

    result = RewardResult()
    result.outage_flags = {sid: False for sid in range(get_num_slices())}
    result.soft_flags = {sid: False for sid in range(get_num_slices())}

    # Extract per-slice metrics (defaults to 0 if missing)
    embb_metrics = metrics.get(0, {})
    urllc_metrics = metrics.get(1, {})
    mtc_metrics = metrics.get(2, {})

    # ---- Normalization parameters ----
    max_achievable = float(config.get(
        "max_achievable_thr_mbps", DEFAULT_MAX_ACHIEVABLE_THR_MBPS
    ))
    bf_norm = float(config.get(
        "bfs_normalization_bytes", sla["urllc_outage_bfs_bytes"]
    ))

    # ---- h_1: eMBB optimization term (Eq. 16) ----
    # Paper: h_1 = (1/|Λ1|) * Σ thr(UE) normalized by max achievable rate
    embb_thr = float(embb_metrics.get("throughputMbps_sum", 0.0))
    embb_ue_count = float(embb_metrics.get("ue_count", 1))
    avg_embb_thr = embb_thr / max(embb_ue_count, 1.0)
    h_1 = min(avg_embb_thr / max_achievable, 1.0) if max_achievable > 0 else 0.0

    # ---- h_2: URLLC optimization term (Eq. 17) ----
    # Paper: h_2 = exp(-max_UE bfs(UE) / normalization)
    urllc_max_bfs = float(urllc_metrics.get("bufferBytes_max", 0.0))
    normalized_bfs = urllc_max_bfs / bf_norm if bf_norm > 0 else urllc_max_bfs
    h_2 = math.exp(-normalized_bfs)

    # ---- h_3: MTC optimization term (Eq. 18) ----
    # Paper: h_3 = (1/|Λ3|) * Σ thr(UE) normalized by max achievable rate
    mtc_thr = float(mtc_metrics.get("throughputMbps_sum", 0.0))
    mtc_ue_count = float(mtc_metrics.get("ue_count", 1))
    avg_mtc_thr = mtc_thr / max(mtc_ue_count, 1.0)
    h_3 = min(avg_mtc_thr / max_achievable, 1.0) if max_achievable > 0 else 0.0

    # ---- Base optimization reward (Eq. 8) ----
    # ropt = alpha*h_1 + beta*h_2 + gamma*h_3 + 1/cost(sch)
    opt_reward = alpha * h_1 + beta * h_2 + gamma_val * h_3

    scheduler_id = action_info.get("scheduler_id", -1) if action_info else -1
    if scheduler_id >= 0:
        scheduler_cost_val = get_scheduler_cost(scheduler_id)
        opt_reward += (1.0 / scheduler_cost_val)

    # ---- Terminal conditions (Eq. 12) ----
    # Outage: -Σ(φ_j * ω_j), terminate
    # Soft: 0 reward, terminate
    # Otherwise: ropt
    #
    # Paper uses probabilistic vrsla_ij per timestep (Eq. 10-11).
    # Since we have 1 sample per timestep (10ms) and reliability=100%,
    # the instantaneous check is equivalent. Consecutive-period
    # confirmation filters simulator jitter (HARQ), which is a practical
    # addition beyond the paper.

    in_warmup = step_count < warmup_steps

    # Evaluate instantaneous conditions
    current_outage_conditions = _evaluate_outage_conditions(metrics, sla, config)
    current_soft_conditions = _evaluate_soft_conditions(metrics, sla)

    # Consecutive-period confirmation for outage
    history = kpi_history if kpi_history is not None else []
    for sid in range(get_num_slices()):
        if current_outage_conditions[sid]:
            streak = 1
            for past_metrics in reversed(history):
                past_conditions = _evaluate_outage_conditions(past_metrics, sla, config)
                if past_conditions[sid]:
                    streak += 1
                else:
                    break
            if streak >= consecutive_outage_steps and not in_warmup:
                result.outage_flags[sid] = True

    # Soft conditions are instantaneous (no consecutive confirmation needed,
    # as over-allocation is deterministic, not jitter-prone)
    if not in_warmup:
        for sid in range(get_num_slices()):
            if current_soft_conditions[sid]:
                result.soft_flags[sid] = True

    outage_slices = [sid for sid, flag in result.outage_flags.items() if flag]
    soft_slices = [sid for sid, flag in result.soft_flags.items() if flag]

    sla_violation = bool(outage_slices or soft_slices)
    terminal_reason = "none"

    if outage_slices:
        # Eq. 12: r = -Σ(φ_j * ω_j), terminal
        weights_map = {0: alpha, 1: beta, 2: gamma_val}
        result.reward = -sum(weights_map[sid] for sid in outage_slices)
        result.terminated = terminate_on_sla_violation
        terminal_reason = "outage"
    elif soft_slices:
        # Eq. 12: r = 0, terminal (soft SLA violation)
        result.reward = 0.0
        result.terminated = terminate_on_sla_violation
        terminal_reason = "soft"
    else:
        result.reward = opt_reward
        result.terminated = False

    result.optimization_terms = {
        "h_1_embb": h_1,
        "h_2_urllc": h_2,
        "h_3_mtc": h_3,
        "opt_reward": opt_reward,
    }

    result.debug_info = {
        "scenario": scenario,
        "reward_mode": reward_mode,
        "embb_throughput": embb_thr,
        "urllc_bufferBytes_max": urllc_max_bfs,
        "mtc_throughput": mtc_thr,
        "mtc_tdp": float(mtc_metrics.get("dLostPackets_sum", 0.0)),
        "outage_slices": outage_slices,
        "soft_slices": soft_slices,
        "sla_violation": sla_violation,
        "terminate_on_sla_violation": terminate_on_sla_violation,
        "terminal_reason": terminal_reason if result.terminated else "none",
        "action_info": action_info or {},
        "avg_embb_thr_per_ue": avg_embb_thr,
        "avg_mtc_thr_per_ue": avg_mtc_thr,
        "normalized_bfs": normalized_bfs,
    }

    resource_terms, resource_debug = _compute_resource_efficiency_terms(
        metrics, sla, action_info, config
    )
    eff_weight = float(config.get("resource_efficiency_weight", 0.20))
    match_weight = float(config.get("need_match_weight", 0.15))
    waste_weight = float(config.get("waste_penalty_weight", 0.25))
    under_weight = float(config.get("under_allocation_penalty_weight", 0.10))
    embb_soft_guard_weight = float(config.get("embb_soft_guard_penalty_weight", 0.25))
    smooth_weight = float(config.get("action_smoothness_weight", 0.05))

    shaping = 0.0
    if reward_mode == "resource_efficient" and not sla_violation:
        shaping = (
            eff_weight * resource_terms["resource_efficiency"]
            + match_weight * resource_terms["need_allocation_match"]
            - waste_weight * resource_terms["over_allocation"]
            - under_weight * resource_terms["under_allocation"]
            - embb_soft_guard_weight * resource_terms["embb_soft_guard_penalty"]
            - smooth_weight * resource_terms["action_smoothness_penalty"]
        )
        result.reward += shaping

    result.optimization_terms.update(resource_terms)
    result.optimization_terms.update({
        "resource_efficient_shaping": shaping,
        "resource_efficient_reward": result.reward,
    })
    result.debug_info.update(resource_terms)
    result.debug_info.update(resource_debug)
    result.debug_info.update({
        "resource_efficiency_weights": {
            "resource_efficiency_weight": eff_weight,
            "need_match_weight": match_weight,
            "waste_penalty_weight": waste_weight,
            "under_allocation_penalty_weight": under_weight,
            "embb_soft_guard_penalty_weight": embb_soft_guard_weight,
            "action_smoothness_weight": smooth_weight,
        },
        "resource_efficient_shaping": shaping,
    })

    return result
