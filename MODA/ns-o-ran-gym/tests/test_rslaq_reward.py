"""Tests for rslaq_reward module (paper-faithful implementation).

Follows equations from:
  Yungaicela-Naula et al., "RSLAQ", IEEE TMC 2026.
  - Eq. 16: h_1 = per-UE avg thr / max achievable rate (eMBB)
  - Eq. 17: h_2 = exp(-max_bfs / normalization) (URLLC)
  - Eq. 18: h_3 = per-UE avg thr / max achievable rate (MTC)
  - Eq. 8:  ropt = alpha*h_1 + beta*h_2 + gamma*h_3 + 1/cost
  - Eq. 12: Terminal conditions (soft and outage)
  - MTC is No-Policy (Table V) → no outage condition
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.rslaq_reward import (
    compute_rslaq_reward,
    get_scheduler_cost,
    SLA_BY_SCENARIO,
    ALPHA,
    BETA,
    GAMMA,
)

# Paper: normalization parameters
MAX_ACHIEVABLE_THR_MBPS = 150.0  # max achievable rate per UE (Mbps)
POST_WARMUP = 10  # step_count past warmup so terminal conditions fire
ZERO_WARMUP_CONFIG = {"warmup_steps": 0}
# Disable consecutive-period confirmation: single step triggers terminal
INSTANT_OUTAGE_CONFIG = {"warmup_steps": 0, "consecutive_outage_steps": 1}


def _good_metrics():
    """Metrics where all slices meet SLA targets (normal scenario)."""
    return {
        0: {
            "throughputMbps_sum": 12.0,
            "dTxBytes_sum": 5000.0,
            "dLostPackets_sum": 100,
            "ue_count": 5.0,
        },
        1: {
            "throughputMbps_sum": 0.0,
            "bufferBytes_max": 500.0,
            "dLostPackets_sum": 0,
            "ue_count": 5.0,
        },
        2: {
            "throughputMbps_sum": 5.0,
            "dLostPackets_sum": 50,
            "ue_count": 10.0,
        },
    }


def _h1_paper(throughput_sum, ue_count, max_rate=MAX_ACHIEVABLE_THR_MBPS):
    """Paper Eq. 16: h_1 = min(avg_per_ue_thr / max_rate, 1.0)."""
    avg = throughput_sum / max(ue_count, 1.0)
    return min(avg / max_rate, 1.0)


def _h2_paper(buffer_max, bf_norm=None):
    """Paper Eq. 17: h_2 = exp(-max_bfs / normalization)."""
    if bf_norm is None:
        bf_norm = SLA_BY_SCENARIO["normal"]["urllc_outage_bfs_bytes"]
    return math.exp(-buffer_max / bf_norm)


def _h3_paper(throughput_sum, ue_count, max_rate=MAX_ACHIEVABLE_THR_MBPS):
    """Paper Eq. 18: h_3 = min(avg_per_ue_thr / max_rate, 1.0)."""
    avg = throughput_sum / max(ue_count, 1.0)
    return min(avg / max_rate, 1.0)


# ── Basic reward tests ──────────────────────────────────────────────

def test_all_slices_meet_sla():
    """Positive reward when all slices meet SLA."""
    result = compute_rslaq_reward(_good_metrics(), scenario="normal", step_count=POST_WARMUP)
    assert result.reward > 0
    assert not result.terminated
    assert not any(result.outage_flags.values())
    assert not any(result.soft_flags.values())


def test_embb_outage_below_min():
    """eMBB outage when throughput below min AND demand exists (Eq. 13)."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 5.0  # below 10 Mbps min
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=INSTANT_OUTAGE_CONFIG)
    assert result.outage_flags[0] is True
    assert result.terminated
    assert result.reward == -ALPHA


def test_demand_aware_embb_outage_disabled_by_default():
    """Paper-faithful default still uses the absolute eMBB SLA threshold."""
    metrics = _good_metrics()
    metrics[0]["dTxBytes_sum"] = 7640.0
    metrics[0]["throughputMbps_sum"] = 6.112
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=INSTANT_OUTAGE_CONFIG)
    assert result.outage_flags[0] is True
    assert result.terminated


def test_demand_aware_embb_outage_suppresses_low_load_false_positive():
    """Demand-aware mode caps the eMBB SLA by offered traffic."""
    metrics = _good_metrics()
    metrics[0]["dTxBytes_sum"] = 7640.0  # 6.112 Mbps over 10 ms
    metrics[0]["throughputMbps_sum"] = 6.112
    config = {
        **INSTANT_OUTAGE_CONFIG,
        "demand_aware_embb_outage": True,
        "period_ms": 10,
    }
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=config)
    assert result.outage_flags[0] is False
    assert not result.terminated


def test_demand_aware_embb_outage_still_flags_unserved_demand():
    """Demand-aware mode still flags eMBB when offered demand is not served."""
    metrics = _good_metrics()
    metrics[0]["dTxBytes_sum"] = 20000.0  # 16 Mbps offered over 10 ms
    metrics[0]["throughputMbps_sum"] = 6.0
    config = {
        **INSTANT_OUTAGE_CONFIG,
        "demand_aware_embb_outage": True,
        "period_ms": 10,
    }
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=config)
    assert result.outage_flags[0] is True
    assert result.terminated


# ── h_2: URLLC exponential formula (Eq. 17) ─────────────────────────

def test_urllc_uses_buffer_max_not_plr():
    """URLLC uses bufferBytes_max (not PLR) for h_2 (Eq. 17)."""
    metrics = _good_metrics()
    metrics[1]["plr_mean"] = 999.0  # should be ignored
    metrics[1]["bufferBytes_max"] = 100.0
    result = compute_rslaq_reward(metrics, scenario="normal")
    expected = _h2_paper(100.0)
    assert abs(result.optimization_terms["h_2_urllc"] - expected) < 1e-9
    assert not result.outage_flags[1]


def test_urllc_outage_buffer_exceeds_threshold():
    """URLLC outage when bufferBytes_max > threshold (Eq. 14)."""
    metrics = _good_metrics()
    metrics[1]["bufferBytes_max"] = 15000.0  # > 10000.0 threshold
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=INSTANT_OUTAGE_CONFIG)
    assert result.outage_flags[1] is True
    assert result.terminated
    assert result.reward == -BETA


def test_urlcc_h2_exponential_formula():
    """h_2 = exp(-max_bfs / bf_norm) (Eq. 17)."""
    metrics = _good_metrics()
    metrics[1]["bufferBytes_max"] = 2500.0
    result = compute_rslaq_reward(metrics, scenario="normal")
    expected = _h2_paper(2500.0)
    assert abs(result.optimization_terms["h_2_urllc"] - expected) < 1e-9


def test_h2_at_zero_buffer():
    """h_2 = 1.0 when buffer is zero (empty buffer = perfect)."""
    metrics = _good_metrics()
    metrics[1]["bufferBytes_max"] = 0.0
    result = compute_rslaq_reward(metrics, scenario="normal")
    expected = _h2_paper(0.0)
    assert abs(expected - 1.0) < 1e-9
    assert abs(result.optimization_terms["h_2_urllc"] - 1.0) < 1e-9


def test_h2_at_threshold():
    """h_2 = exp(-1) ≈ 0.368 at outage threshold."""
    bf_norm = SLA_BY_SCENARIO["normal"]["urllc_outage_bfs_bytes"]
    metrics = _good_metrics()
    metrics[1]["bufferBytes_max"] = bf_norm
    result = compute_rslaq_reward(metrics, scenario="normal")
    expected = _h2_paper(bf_norm, bf_norm)
    assert abs(expected - math.exp(-1.0)) < 1e-9
    assert abs(result.optimization_terms["h_2_urllc"] - expected) < 1e-9


# ── h_1, h_3: throughput normalization (Eq. 16, 18) ─────────────────

def test_embb_h1_per_ue_normalized():
    """h_1 = min(avg_per_ue_thr / max_achievable_rate, 1.0) (Eq. 16)."""
    metrics = _good_metrics()
    metrics[0]["ue_count"] = 5.0
    metrics[0]["throughputMbps_sum"] = 30.0  # 6 Mbps per UE
    result = compute_rslaq_reward(metrics, scenario="normal")
    expected = _h1_paper(30.0, 5.0)
    assert abs(result.optimization_terms["h_1_embb"] - expected) < 1e-9
    assert expected < 1.0  # 6/150 = 0.04

    # max throughput cap
    metrics[0]["throughputMbps_sum"] = 1000.0  # 200 Mbps per UE
    result = compute_rslaq_reward(metrics, scenario="normal")
    assert result.optimization_terms["h_1_embb"] == 1.0


def test_mtc_h3_per_ue_normalized():
    """h_3 = min(avg_per_ue_thr / max_achievable_rate, 1.0) (Eq. 18)."""
    metrics = _good_metrics()
    metrics[2]["ue_count"] = 10.0
    metrics[2]["throughputMbps_sum"] = 150.0  # 15 Mbps per UE
    result = compute_rslaq_reward(metrics, scenario="normal")
    expected = _h3_paper(150.0, 10.0)
    assert abs(result.optimization_terms["h_3_mtc"] - expected) < 1e-9
    assert expected < 1.0  # 15/150 = 0.1

    # max cap
    metrics[2]["throughputMbps_sum"] = 2000.0
    result = compute_rslaq_reward(metrics, scenario="normal")
    assert result.optimization_terms["h_3_mtc"] == 1.0


# ── MTC No-Policy (Table V) ─────────────────────────────────────────

def test_mtc_no_policy_no_outage():
    """MTC is No-Policy → never triggers outage (Table V)."""
    metrics = _good_metrics()
    metrics[2]["dLostPackets_sum"] = 5000.0  # would have been outage
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=INSTANT_OUTAGE_CONFIG)
    assert not result.outage_flags[2]
    assert not result.terminated
    assert result.reward > 0  # still gets optimization reward


# ── Soft SLA conditions (Eq. 12, 15) ────────────────────────────────

def test_embb_soft_violation_triggers():
    """eMBB exceeding soft_max triggers soft SLA violation (Eq. 12, 15)."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 30.0  # > soft_max (15.0 for normal)
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=INSTANT_OUTAGE_CONFIG)
    assert result.soft_flags[0] is True
    assert result.terminated
    assert result.reward == 0.0


def test_embb_below_soft_max_no_violation():
    """eMBB below soft_max → no soft violation."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 12.0  # < soft_max (15.0)
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP)
    assert not result.soft_flags[0]
    assert not result.terminated
    assert result.reward > 0


# ── Outage penalty (Eq. 12) ─────────────────────────────────────────

def test_outage_reward_negative_weighted_sum():
    """Outage reward = -sum(phi_j * omega_j) (Eq. 12)."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 5.0     # eMBB outage
    metrics[1]["bufferBytes_max"] = 15000.0     # URLLC outage
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=INSTANT_OUTAGE_CONFIG)
    assert result.outage_flags[0] is True
    assert result.outage_flags[1] is True
    assert not result.outage_flags[2]  # MTC is no-policy
    expected_penalty = -(ALPHA + BETA)
    assert abs(result.reward - expected_penalty) < 1e-9
    assert result.terminated


# ── Scheduler cost (Eq. 8, 9) ───────────────────────────────────────

def test_scheduler_cost():
    """Scheduler cost mapping (Eq. 9): RR=1, PF=2, BCQI=2."""
    assert get_scheduler_cost(0) == 1.0  # RR
    assert get_scheduler_cost(1) == 2.0  # PF
    assert get_scheduler_cost(2) == 2.0  # BCQI
    assert get_scheduler_cost(-1) == 1.0  # no scheduler
    assert get_scheduler_cost(99) == 1.0  # unknown


def test_scheduler_cost_adds_not_multiplies():
    """Scheduler cost term is added (not multiplied) (Eq. 8)."""
    metrics = _good_metrics()

    result_no_scheduler = compute_rslaq_reward(
        metrics, scenario="normal", action_info=None
    )
    result_rr = compute_rslaq_reward(
        metrics, scenario="normal", action_info={"scheduler_id": 0}
    )
    result_pf = compute_rslaq_reward(
        metrics, scenario="normal", action_info={"scheduler_id": 1}
    )

    # RR cost=1.0 → +1/1.0 = +1.0
    diff_rr = result_rr.reward - result_no_scheduler.reward
    assert abs(diff_rr - 1.0) < 1e-9

    # PF cost=2.0 → +1/2.0 = +0.5
    diff_pf = result_pf.reward - result_no_scheduler.reward
    assert abs(diff_pf - 0.5) < 1e-9


def test_no_scheduler_no_cost_term():
    """Without scheduler_id, no cost term is added."""
    metrics = _good_metrics()

    result_no_action = compute_rslaq_reward(metrics, scenario="normal")
    result_no_scheduler = compute_rslaq_reward(
        metrics, scenario="normal", action_info={"scheduler_id": -1}
    )

    assert abs(result_no_action.reward - result_no_scheduler.reward) < 1e-9

    terms = result_no_action.optimization_terms
    expected = ALPHA * terms["h_1_embb"] + BETA * terms["h_2_urllc"] + GAMMA * terms["h_3_mtc"]
    assert abs(result_no_action.reward - expected) < 1e-9



# ── Resource-efficient contribution ─────────────────────────────────

def _resource_efficiency_metrics():
    """Metrics with dominant URLLC demand and no SLA violation."""
    metrics = _good_metrics()
    metrics[0].update({
        "dTxBytes_sum": 1000.0,
        "dRxBytes_sum": 1000.0,
        "bufferBytes_max": 0.0,
        "dLostPackets_sum": 0.0,
        "throughputMbps_sum": 12.0,
    })
    metrics[1].update({
        "dTxBytes_sum": 90000.0,
        "dRxBytes_sum": 90000.0,
        "bufferBytes_max": 1000.0,
        "dLostPackets_sum": 0.0,
    })
    metrics[2].update({
        "dTxBytes_sum": 1000.0,
        "dRxBytes_sum": 1000.0,
        "bufferBytes_max": 0.0,
        "dLostPackets_sum": 0.0,
        "throughputMbps_sum": 5.0,
    })
    return metrics


def test_paper_reward_mode_preserves_default():
    """Explicit paper mode keeps the default RSLAQ reward unchanged."""
    default = compute_rslaq_reward(_good_metrics(), scenario="normal")
    paper = compute_rslaq_reward(
        _good_metrics(), scenario="normal", config={"reward_mode": "paper"}
    )
    assert abs(default.reward - paper.reward) < 1e-12
    assert default.optimization_terms == paper.optimization_terms


def test_paper_reward_mode_exposes_resource_efficiency_diagnostics():
    """Paper mode logs resource-efficiency metrics without shaping reward."""
    metrics = _resource_efficiency_metrics()
    result = compute_rslaq_reward(
        metrics,
        scenario="normal",
        action_info={"prb_pct": [5.0, 90.0, 5.0]},
        config={
            "reward_mode": "paper",
            "resource_dynamic_need_weight": 1.0,
            "resource_waste_deadband": 0.0,
        },
    )

    assert result.reward == result.optimization_terms["opt_reward"]
    assert result.optimization_terms["resource_efficient_shaping"] == 0.0
    for key in (
        "resource_efficiency",
        "need_allocation_match",
        "over_allocation",
        "under_allocation",
    ):
        assert key in result.optimization_terms
        assert key in result.debug_info
    assert result.optimization_terms["need_allocation_match"] > 0.0


def test_resource_efficient_penalizes_over_allocation():
    """The contribution rewards PRB shares that follow active slice need."""
    metrics = _resource_efficiency_metrics()
    config = {
        "reward_mode": "resource_efficient",
        "resource_dynamic_need_weight": 1.0,
        "resource_waste_deadband": 0.0,
        "warmup_steps": 0,
    }
    matched = compute_rslaq_reward(
        metrics,
        scenario="normal",
        action_info={"prb_pct": [5.0, 90.0, 5.0]},
        config=config,
        step_count=POST_WARMUP,
    )
    wasteful = compute_rslaq_reward(
        metrics,
        scenario="normal",
        action_info={"prb_pct": [90.0, 5.0, 5.0]},
        config=config,
        step_count=POST_WARMUP,
    )
    assert matched.reward > wasteful.reward
    assert matched.optimization_terms["need_allocation_match"] > wasteful.optimization_terms["need_allocation_match"]
    assert wasteful.optimization_terms["over_allocation"] > matched.optimization_terms["over_allocation"]


def test_resource_efficient_penalizes_embb_near_soft_limit():
    """The contribution discourages eMBB over-serving before soft SLA zeroes reward."""
    config = {
        "reward_mode": "resource_efficient",
        "resource_efficiency_weight": 0.0,
        "need_match_weight": 0.0,
        "waste_penalty_weight": 0.0,
        "under_allocation_penalty_weight": 0.0,
        "action_smoothness_weight": 0.0,
        "embb_soft_guard_penalty_weight": 1.0,
        "embb_soft_guard_ratio": 0.8,
        "warmup_steps": 0,
    }
    safe_metrics = _good_metrics()
    safe_metrics[0].update({
        "throughputMbps_sum": 12.0,
        "dTxBytes_sum": 20000.0,
        "dRxBytes_sum": 20000.0,
    })
    near_soft_metrics = _good_metrics()
    near_soft_metrics[0].update({
        "throughputMbps_sum": 14.5,
        "dTxBytes_sum": 20000.0,
        "dRxBytes_sum": 20000.0,
    })

    safe = compute_rslaq_reward(
        safe_metrics,
        scenario="normal",
        action_info={"prb_pct": [70.0, 20.0, 10.0]},
        config=config,
        step_count=POST_WARMUP,
    )
    near_soft = compute_rslaq_reward(
        near_soft_metrics,
        scenario="normal",
        action_info={"prb_pct": [70.0, 20.0, 10.0]},
        config=config,
        step_count=POST_WARMUP,
    )

    assert safe.optimization_terms["embb_soft_guard_penalty"] == 0.0
    assert near_soft.optimization_terms["embb_soft_guard_penalty"] > 0.0
    assert near_soft.reward < safe.reward


def test_resource_efficient_preserves_outage_terminal_reward():
    """Resource-efficient diagnostics do not override Eq. 12 outage penalties."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 5.0
    config = {
        **INSTANT_OUTAGE_CONFIG,
        "reward_mode": "resource_efficient",
        "resource_efficiency_weight": 0.0,
        "need_match_weight": 1.0,
        "waste_penalty_weight": 0.0,
        "under_allocation_penalty_weight": 0.0,
        "action_smoothness_weight": 0.0,
    }
    result = compute_rslaq_reward(
        metrics,
        scenario="normal",
        action_info={"prb_pct": [90.0, 5.0, 5.0]},
        config=config,
        step_count=POST_WARMUP,
    )
    assert result.outage_flags[0] is True
    assert result.terminated
    assert result.optimization_terms["resource_efficient_shaping"] == 0.0
    assert result.optimization_terms["need_allocation_match"] > 0.0
    assert result.reward == -ALPHA


def test_resource_efficient_debug_terms():
    """The new reward exposes efficiency terms for campaign analysis."""
    result = compute_rslaq_reward(
        _resource_efficiency_metrics(),
        scenario="normal",
        action_info={"prb_pct": [5.0, 90.0, 5.0]},
        config={"reward_mode": "resource_efficient"},
    )
    for key in (
        "resource_efficiency",
        "need_allocation_match",
        "over_allocation",
        "under_allocation",
        "resource_efficient_shaping",
    ):
        assert key in result.optimization_terms
        assert key in result.debug_info
    assert result.debug_info["reward_mode"] == "resource_efficient"



def test_sla_violation_can_penalize_without_terminating():
    """DDQN can keep the paper Eq. 12 penalty while using ntsr for reset."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 5.0
    config = {
        **INSTANT_OUTAGE_CONFIG,
        "terminate_on_sla_violation": False,
    }
    result = compute_rslaq_reward(
        metrics, scenario="normal", step_count=POST_WARMUP, config=config
    )
    assert result.outage_flags[0] is True
    assert result.reward == -ALPHA
    assert not result.terminated
    assert result.debug_info["sla_violation"] is True
    assert result.debug_info["terminate_on_sla_violation"] is False


def test_resource_efficient_preserves_nonterminal_sla_penalty():
    """Resource-efficient diagnostics keep Eq. 12 penalties without reset."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 5.0
    config = {
        **INSTANT_OUTAGE_CONFIG,
        "reward_mode": "resource_efficient",
        "terminate_on_sla_violation": False,
        "resource_efficiency_weight": 0.0,
        "need_match_weight": 1.0,
        "waste_penalty_weight": 0.0,
        "under_allocation_penalty_weight": 0.0,
        "action_smoothness_weight": 0.0,
    }
    result = compute_rslaq_reward(
        metrics,
        scenario="normal",
        action_info={"prb_pct": [90.0, 5.0, 5.0]},
        config=config,
        step_count=POST_WARMUP,
    )
    assert result.outage_flags[0] is True
    assert not result.terminated
    assert result.optimization_terms["resource_efficient_shaping"] == 0.0
    assert result.optimization_terms["need_allocation_match"] > 0.0
    assert result.reward == -ALPHA


# ── Metadata tests ───────────────────────────────────────────────────

def test_optimization_terms_keys():
    """optimization_terms dict has all expected keys."""
    result = compute_rslaq_reward(_good_metrics(), scenario="normal")
    for key in ("h_1_embb", "h_2_urllc", "h_3_mtc", "opt_reward"):
        assert key in result.optimization_terms


def test_reward_debug_info():
    """debug_info contains scenario and per-slice metrics."""
    result = compute_rslaq_reward(_good_metrics(), scenario="stressed")
    assert result.debug_info["scenario"] == "stressed"
    assert "urllc_bufferBytes_max" in result.debug_info
    assert "mtc_tdp" in result.debug_info
    assert "avg_embb_thr_per_ue" in result.debug_info


# ── Warmup guard tests ───────────────────────────────────────────────

def test_warmup_suppresses_outage():
    """Terminal conditions suppressed during warmup (step_count < warmup_steps)."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 5.0  # would be outage
    result = compute_rslaq_reward(metrics, scenario="normal", step_count=0)
    assert not result.outage_flags[0]
    assert not result.terminated


def test_post_warmup_allows_outage():
    """Terminal conditions active after warmup period."""
    metrics = _good_metrics()
    metrics[0]["throughputMbps_sum"] = 5.0
    result = compute_rslaq_reward(metrics, scenario="normal",
                                  step_count=POST_WARMUP, config=INSTANT_OUTAGE_CONFIG)
    assert result.outage_flags[0] is True


def test_consecutive_outage_needs_streak():
    """Single-step outage condition doesn't trigger without consecutive streak."""
    metrics = _good_metrics()
    metrics[1]["bufferBytes_max"] = 15000.0
    result = compute_rslaq_reward(
        metrics, scenario="normal",
        step_count=POST_WARMUP, config=ZERO_WARMUP_CONFIG,
    )
    # With empty kpi_history, streak=1 < consecutive_outage_steps (default 5)
    assert not result.outage_flags[1]


def test_consecutive_outage_with_history():
    """5 consecutive outage steps trigger the flag."""
    metrics = _good_metrics()
    metrics[1]["bufferBytes_max"] = 15000.0
    history = []
    for _ in range(4):
        history.append({
            0: {"throughputMbps_sum": 12.0, "dTxBytes_sum": 5000.0},
            1: {"bufferBytes_max": 15000.0},
            2: {"throughputMbps_sum": 5.0, "dLostPackets_sum": 50},
        })
    result = compute_rslaq_reward(
        metrics, scenario="normal", step_count=POST_WARMUP,
        config=ZERO_WARMUP_CONFIG, kpi_history=history,
    )
    assert result.outage_flags[1] is True
    assert result.terminated


# ── Run manually ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_all_slices_meet_sla()
    test_embb_outage_below_min()
    test_urllc_uses_buffer_max_not_plr()
    test_urllc_outage_buffer_exceeds_threshold()
    test_urlcc_h2_exponential_formula()
    test_h2_at_zero_buffer()
    test_h2_at_threshold()
    test_embb_h1_per_ue_normalized()
    test_mtc_h3_per_ue_normalized()
    test_mtc_no_policy_no_outage()
    test_embb_soft_violation_triggers()
    test_embb_below_soft_max_no_violation()
    test_outage_reward_negative_weighted_sum()
    test_scheduler_cost()
    test_scheduler_cost_adds_not_multiplies()
    test_no_scheduler_no_cost_term()
    test_optimization_terms_keys()
    test_reward_debug_info()
    test_warmup_suppresses_outage()
    test_post_warmup_allows_outage()
    test_consecutive_outage_needs_streak()
    test_consecutive_outage_with_history()
    test_paper_reward_mode_preserves_default()
    test_paper_reward_mode_exposes_resource_efficiency_diagnostics()
    test_resource_efficient_penalizes_over_allocation()
    test_resource_efficient_penalizes_embb_near_soft_limit()
    test_resource_efficient_preserves_outage_terminal_reward()
    test_resource_efficient_debug_terms()
    test_sla_violation_can_penalize_without_terminating()
    test_resource_efficient_preserves_nonterminal_sla_penalty()
    print("All reward tests passed.")
