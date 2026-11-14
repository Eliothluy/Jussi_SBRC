"""Joint state, normalization and window-alignment tests (§11–§13/§23).

Covers: test_no_future_information, test_joint_feature_vector,
test_normalization, test_decision_window_alignment, scaler freeze rules
(§13) and the manifest round-trip.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from nsoran.sqr.sqr_kpis import SliceKpis, SqrKpmWindow, UeKpmRow
from nsoran.sqr_jc.feature_scaler import FeatureScaler
from nsoran.sqr_jc.joint_state import (
    DECISION_PERIOD_MS,
    FEATURE_INDEX,
    JOINT_STATE_DIM,
    JointState,
    decision_window_of_stamp,
    governed_outcome_window,
    kpm_stamp_ms,
    state_from_kpm_window,
    urllc_delay_window_stats,
    window_of_stamp,
)
from nsoran.sqr_jc.sla_margin import SLA_EMBB_MBPS, SLA_URLLC_MS
from nsoran.sqr_jc.yaml_subset import dump, load


def _row(ts: int, sid: int, thr: float, queue: float = 0.0, snr: float = 20.0,
         tx_pk: float = 0.0, rx_pk: float = 0.0) -> UeKpmRow:
    return UeKpmRow(
        timestamp=ts, ue_imsi=sid + 1, slice_id=sid, d_tx_bytes=1000.0,
        d_rx_bytes=900.0, plr=0.0, resource_share_pct=10.0, d_lost_packets=0.0,
        throughput_mbps=thr, buffer_bytes=queue, window_throughput_mbps=thr,
        snr_db=snr, snr_was_imputed=False, snr_imputation=None,
        queue_bytes=queue, d_tx_packets=tx_pk, d_rx_packets=rx_pk,
        d_delay_sum_sec=0.0, cqi_wideband=10,
    )


def _window(ts: int, rows: list[UeKpmRow]) -> SqrKpmWindow:
    by_slice: dict[int, list] = {0: [], 1: [], 2: []}
    for r in rows:
        by_slice[r.slice_id].append(r)
    slices = {}
    for sid in range(3):
        rs = by_slice[sid]
        thr = [r.window_throughput_mbps for r in rs]
        slices[sid] = SliceKpis(
            slice_id=sid, n_ues=len(rs),
            lowest_thr_mbps=min(thr) if thr else None,
            p5_thr_mbps=min(thr) if thr else None,
            mean_thr_mbps=(sum(thr) / len(thr)) if thr else None,
            queue_bytes_sum=sum(r.queue_bytes for r in rs),
            violation_rate=None, n_snr_imputed=0, snr_imputations={},
            snr_values=tuple(r.snr_db for r in rs),
        )
    return SqrKpmWindow(timestamp=ts, slices=slices, total_ues=len(rows),
                        cell_snr_median=20.0, rows=tuple(rows))


# ---------------------------------------------------------------------------
# test_no_future_information (§12)
# ---------------------------------------------------------------------------

def test_no_future_information():
    """Every JointState slot is pre-decision (§11); the class has NO slot
    that could carry the outcome of the governed window, so future info
    cannot be attached; the governed window is strictly the NEXT one."""
    st = JointState(n_e=1, n_u=2, n_m=0, window_index=41)
    assert governed_outcome_window(st.window_index) == 42
    # the class exposes no future/outcome field
    forbidden = {"outcome_kpi", "future_kpi", "kpi_t_plus_1", "result",
                 "outcome_margin", "feasibility_class"}
    for name in forbidden:
        assert not hasattr(st, name), f"future-information slot {name!r} exists"
    # setting unknown attributes is possible on a plain dataclass — so the
    # FEATURE path must ignore them: raw_vector() has fixed dim 23 built
    # ONLY from the documented pre-decision fields.
    st.__dict__["outcome_kpi"] = 999.0  # hostile attach
    raw = st.raw_vector()
    assert raw.shape == (JOINT_STATE_DIM,)
    assert 999.0 not in raw
    # builder consumes only the closed window + explicit history
    w = _window(4300, [_row(4300, 0, 4.0, queue=1e5)])
    st2 = state_from_kpm_window(w, window_index=42)
    assert st2.n_e == 1 and st2.n_u == 0 and st2.n_m == 0
    assert st2.kpi_e_raw is None  # no KPI injected -> no future leak


def test_no_future_information_in_builder_inputs():
    """The builder's inputs are the closed window and PREVIOUS aggregates —
    the governed window's KPI stamp lies strictly in the future of the
    decision (alignment documented in §23 and tested separately)."""
    w = _window(4200, [_row(4200, 1, 0.0)])
    st = state_from_kpm_window(
        w,
        closed_window_kpis={"urllc_p95_ms": 8.0},  # KPI OF THE CLOSED WINDOW
        window_index=42,
    )
    assert st.kpi_u_raw == 8.0                      # available at decision time
    assert kpm_stamp_ms(governed_outcome_window(42)) == 4400  # future stamp
    assert kpm_stamp_ms(42) == 4300                 # the closed window's stamp


# ---------------------------------------------------------------------------
# test_joint_feature_vector (§5: x = [z_joint, a/C])
# ---------------------------------------------------------------------------

def test_joint_feature_vector():
    from nsoran.sqr_jc.feature_scaler import FeatureScaler

    rng = np.random.default_rng(7)
    # build explicitly in the documented feature order:
    data = np.column_stack([
        rng.uniform(0, 10, 200), rng.uniform(0, 10, 200), rng.uniform(0, 10, 200),
        rng.uniform(0, 2e6, 200), rng.uniform(0, 2e6, 200), rng.uniform(0, 2e6, 200),
        rng.uniform(0, 10, 200), rng.uniform(0, 20, 200), rng.uniform(0.5, 1, 200),
        rng.uniform(0, 30, 200), rng.uniform(0, 1, 200), rng.uniform(0, 1, 200),
        rng.uniform(0, 30, 200), rng.uniform(0, 30, 200), rng.uniform(0, 30, 200),
        rng.uniform(0, 53, 200), rng.uniform(0, 53, 200), rng.uniform(0, 53, 200),
        rng.integers(0, 2, 200), rng.integers(0, 2, 200), rng.integers(0, 2, 200),
        rng.uniform(0, 1, 200), rng.uniform(0, 20, 200),
    ]).astype(float)
    assert data.shape[1] == JOINT_STATE_DIM
    scaler = FeatureScaler(c=53)
    scaler.fit(data)

    st = JointState(n_e=2, n_u=1, n_m=3, queue_e=1e6, queue_u=1000.0, queue_m=500.0,
                    kpi_e_raw=7.0, kpi_u_raw=5.0, kpi_m_raw=1.0,
                    offered_e=8.0, offered_u=0.2, offered_m=0.002,
                    snr_e=15.0, snr_u=15.0, snr_m=15.0,
                    prev_a_e=18, prev_a_u=13, prev_a_m=3,
                    viol_e=1, viol_u=0, viol_m=0,
                    urllc_miss_raw=0.04, urllc_p99_raw=12.0)
    x = st.feature_x((20, 10, 5), c=53, scaler=scaler)
    assert x.shape == (JOINT_STATE_DIM + 3,)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    # action block = a/C exactly
    assert x[-3:] == pytest.approx(np.array([20 / 53, 10 / 53, 5 / 53]))
    # KPI eMBB feature = clip(7.0/3.5) = 1.0 (saturated)
    assert x[FEATURE_INDEX["kpi_e"]] == pytest.approx(1.0)
    # URLLC p95 feature = 5/10 = 0.5
    assert x[FEATURE_INDEX["kpi_u"]] == pytest.approx(0.5)
    # allocation features
    assert x[FEATURE_INDEX["prev_a_e"]] == pytest.approx(18 / 53)


# ---------------------------------------------------------------------------
# test_normalization (§13 rules; CAL p1/p99 freeze)
# ---------------------------------------------------------------------------

def test_normalization_physical_rules():
    scaler = FeatureScaler(c=53)
    # KPI / SLA clipping
    assert scaler.transform_feature("kpi_e", 7.0) == pytest.approx(1.0)
    assert scaler.transform_feature("kpi_e", 3.5) == pytest.approx(1.0)
    assert scaler.transform_feature("kpi_u", 20.0) == pytest.approx(1.0)
    assert scaler.transform_feature("kpi_u", 5.0) == pytest.approx(0.5)
    # PDR natural
    assert scaler.transform_feature("kpi_m", 0.95) == pytest.approx(0.95)
    # SNR R09 scale (audited inheritance)
    assert scaler.transform_feature("snr_e", 30.0) == pytest.approx(1.0)
    assert scaler.transform_feature("snr_u", 15.0) == pytest.approx(0.5)
    # allocation /C
    assert scaler.transform_feature("prev_a_m", 53) == pytest.approx(1.0)
    assert scaler.transform_feature("prev_a_u", 13) == pytest.approx(13 / 53)
    # miss rate natural
    assert scaler.transform_feature("urllc_miss", 0.05) == pytest.approx(0.05)


def test_normalization_cal_p1p99_freeze(tmp_path):
    """§13: counts/queues/offered -> CAL p1/p99 clip -> [0,1], FROZEN;
    refit after freeze raises; the manifest round-trips."""
    rng = np.random.default_rng(3)
    data = np.column_stack([
        rng.uniform(0, 10, 500), rng.uniform(0, 10, 500), rng.uniform(0, 10, 500),
        rng.uniform(0, 2e6, 500), rng.uniform(0, 2e6, 500), rng.uniform(0, 2e6, 500),
        rng.uniform(0, 10, 500), rng.uniform(0, 20, 500), rng.uniform(0.5, 1, 500),
        rng.uniform(0, 30, 500), rng.uniform(0, 1, 500), rng.uniform(0, 1, 500),
        rng.uniform(0, 30, 500), rng.uniform(0, 30, 500), rng.uniform(0, 30, 500),
        rng.uniform(0, 53, 500), rng.uniform(0, 53, 500), rng.uniform(0, 53, 500),
        rng.integers(0, 2, 500), rng.integers(0, 2, 500), rng.integers(0, 2, 500),
        rng.uniform(0, 1, 500), rng.uniform(0, 20, 500),
    ]).astype(float)
    scaler = FeatureScaler(c=53)
    scaler.fit(data)
    n_e_raw = 7.0
    p1 = scaler.to_manifest()["n_e_p1"]
    p99 = scaler.to_manifest()["n_e_p99"]
    expected = min(max((n_e_raw - p1) / (p99 - p1), 0.0), 1.0)
    assert scaler.transform_feature("n_e", n_e_raw) == pytest.approx(expected)

    path = str(tmp_path / "feature_scaler_manifest.yaml")
    scaler.save_manifest(path, source="CAL")
    frozen = FeatureScaler.load_manifest(path, freeze=True)
    assert frozen.transform_feature("n_e", n_e_raw) == pytest.approx(expected)
    assert frozen.n_cal_samples == 500
    with pytest.raises(RuntimeError):
        frozen.fit(data)  # §13: never refit with PILOT/MAIN
    # the YAML subset parser understands the manifest it wrote
    parsed = load(open(path).read())
    assert parsed["format"].startswith("sqrjc_feature_scaler_manifest")
    assert parsed["source"] == "CAL"
    assert dump(parsed) == open(path).read()  # deterministic emit


# ---------------------------------------------------------------------------
# test_decision_window_alignment (§23)
# ---------------------------------------------------------------------------

def test_decision_window_alignment():
    """Action k governs outcome window k+1, KPI stamp (k+2)*100 ms — the
    frozen replay rule (adjudication.py:343; sqr-sim.cc:1545-1553)."""
    assert DECISION_PERIOD_MS == 100
    assert governed_outcome_window(1263) == 1264
    assert kpm_stamp_ms(1264) == 126500
    # inverse mappings
    assert window_of_stamp(126500) == 1264
    assert decision_window_of_stamp(126500) == 1263
    # the 2.4-II-d replay case: window 1263 -> stamp (1263+2)*100 = 126500
    st = JointState(window_index=1263)
    assert governed_outcome_window(st.window_index) == 1264


def test_urllc_delay_window_stats():
    """Arrival-window grouping of packet delays (frozen dataset.py rule)."""
    rows = [
        {"txSec": 12.5971, "delaySec": 0.005},   # win floor(125.971)=125
        {"txSec": 12.5991, "delaySec": 0.012},   # win 125
        {"txSec": 12.6001, "delaySec": 0.006},   # win 126
    ]
    s125 = urllc_delay_window_stats(rows, 125)
    assert s125["n_pkts"] == 2
    assert s125["miss_rate"] == pytest.approx(0.5)   # 12 ms > 10 ms deadline
    assert s125["p95_ms"] == pytest.approx(11.65)    # linear interpolation
    s126 = urllc_delay_window_stats(rows, 126)
    assert s126["n_pkts"] == 1 and s126["miss_rate"] == 0.0
    assert urllc_delay_window_stats(rows, 999) == {}
