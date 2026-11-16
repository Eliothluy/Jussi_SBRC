"""Controller, bootstrap, SQR-HET and logging tests (§27/§29/§45/§50–§57).

Covers: test_deterministic_tie_break, test_graceful_degradation,
test_shared_bootstrap_determinism, test_replay_state_identity, the §45
no-post-hoc-margin invariant, SQR margins (test_sla_margin_*), §56 column
conformance and the SQR-HET adapter isolation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from nsoran.sqr.sqr_controller import SqrSliceController
from nsoran.sqr_jc.feature_scaler import FeatureScaler
from nsoran.sqr_jc.joint_controller import (
    BOOTSTRAP_SECONDS,
    JointController,
)
from nsoran.sqr_jc.joint_state import JointState
from nsoran.sqr_jc.kernel_models import MultiQuantileKernelModel
from nsoran.sqr_jc.logging import (
    DECISION_COLUMNS,
    KERNEL_LOG_COLUMNS,
    OUTCOME_COLUMNS,
    DecisionLogWriter,
    format_row,
)
from nsoran.sqr_jc.sla_margin import (
    SLA_EMBB_MBPS,
    SLA_MMTC_PDR,
    SLA_URLLC_MS,
    SliceOutcome,
    embb_margin,
    embb_sla_ok,
    mmtc_margin,
    mmtc_sla_ok,
    severity,
    urllc_margin,
    urllc_margin_miss_secondary,
    urllc_sla_ok,
)

DIM = 26
ZDIM = 23


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scaler() -> FeatureScaler:
    rng = np.random.default_rng(5)
    data = np.column_stack([
        rng.uniform(0, 10, 150), rng.uniform(0, 10, 150), rng.uniform(0, 10, 150),
        rng.uniform(0, 2e6, 150), rng.uniform(0, 2e6, 150), rng.uniform(0, 2e6, 150),
        rng.uniform(0, 10, 150), rng.uniform(0, 20, 150), rng.uniform(0.5, 1, 150),
        rng.uniform(0, 30, 150), rng.uniform(0, 1, 150), rng.uniform(0, 1, 150),
        rng.uniform(0, 30, 150), rng.uniform(0, 30, 150), rng.uniform(0, 30, 150),
        rng.uniform(0, 53, 150), rng.uniform(0, 53, 150), rng.uniform(0, 53, 150),
        rng.integers(0, 2, 150), rng.integers(0, 2, 150), rng.integers(0, 2, 150),
        rng.uniform(0, 1, 150), rng.uniform(0, 20, 150),
    ]).astype(float)
    s = FeatureScaler(c=53)
    s.fit(data)
    return s


def _state(**kw) -> JointState:
    base = dict(n_e=2, n_u=1, n_m=2, queue_e=1e5, queue_u=1e3, queue_m=5e3,
                kpi_e_raw=3.0, kpi_u_raw=8.0, kpi_m_raw=0.99,
                prev_a_e=18, prev_a_u=13, prev_a_m=3, window_index=100)
    base.update(kw)
    return JointState(**base)


def _trained_controller(sigma_support: float = 0.85):
    """Controller whose models saw outcomes: eMBB margin responds positively
    to more a_e; URLLC/mMTC mildly positive near the trained region."""
    scaler = _make_scaler()
    models = tuple(MultiQuantileKernelModel(dim=DIM, memory_size=100)
                   for _ in range(3))
    jc = JointController(models, scaler, sigma_support=sigma_support, c=53,
                         bootstrap_seconds=0.0)
    rng = np.random.default_rng(17)
    for k in range(60):
        ae = int(rng.integers(5, 50))
        st = _state(n_e=2, queue_e=5e4, kpi_e_raw=3.5,
                    prev_a_e=ae, prev_a_u=13, prev_a_m=3, window_index=100 + k)
        action = (ae, 13, 3)
        out = SliceOutcome(
            embb_throughput_mbps=3.5 * (1.0 + 0.010 * ae),   # > SLA: margin > 0
            urllc_p95_ms=8.0,                                 # < 10: satisfied
            mmtc_pdr_rolling=0.99,                            # ≥ 0.95: satisfied
        )
        jc.updater.observe_outcome(st, action, out)
    return jc


# ---------------------------------------------------------------------------
# test_sla_margin_* (§7 formulas on the frozen thresholds)
# ---------------------------------------------------------------------------

def test_sla_margin_embb():
    m = embb_margin(7.0)    # 2x the SLA
    assert m == pytest.approx((7.0 - 3.5) / 3.5)
    assert embb_margin(SLA_EMBB_MBPS) == pytest.approx(0.0)
    assert embb_margin(0.0) == pytest.approx(-1.0)
    assert embb_margin(None) is None
    assert embb_sla_ok(None) is True            # vacuous = satisfied
    assert embb_sla_ok(3.49) is False
    assert embb_sla_ok(3.5) is True


def test_sla_margin_urllc():
    assert urllc_margin(5.0) == pytest.approx(0.5)
    assert urllc_margin(SLA_URLLC_MS) == pytest.approx(0.0)
    assert urllc_margin(20.0) == pytest.approx(-1.0)
    assert urllc_margin(None) is None
    assert urllc_sla_ok(None) is True
    assert urllc_sla_ok(10.0) is True           # boundary belongs to OK
    assert urllc_sla_ok(10.1) is False
    # the SECONDARY miss variant exists but is not the primary
    assert urllc_margin_miss_secondary(5.0) == pytest.approx(0.0)
    assert urllc_margin_miss_secondary(0.0) == pytest.approx(1.0)


def test_sla_margin_mmtc():
    assert mmtc_margin(1.0) == pytest.approx((1.0 - 0.95) / 0.95)
    assert mmtc_margin(SLA_MMTC_PDR) == pytest.approx(0.0)
    assert mmtc_margin(None) is None
    assert mmtc_sla_ok(None) is True
    assert mmtc_sla_ok(0.94) is False
    assert mmtc_sla_ok(0.95) is True


def test_severity_rule_63():
    assert severity(-0.25) == pytest.approx(0.25)
    assert severity(0.3) == 0.0
    assert severity(None) == 0.0


def test_slice_outcome_gate_ii_d_semantics():
    out = SliceOutcome(embb_throughput_mbps=4.0, urllc_p95_ms=9.0,
                       mmtc_pdr_rolling=0.97)
    assert out.sla_ok == (True, True, True) and out.joint_ok
    out_bad = SliceOutcome(embb_throughput_mbps=3.0, urllc_p95_ms=11.0,
                           mmtc_pdr_rolling=0.90)
    assert out_bad.sla_ok == (False, False, False)


# ---------------------------------------------------------------------------
# test_deterministic_tie_break (§50/§51/§52 key 5)
# ---------------------------------------------------------------------------

def test_deterministic_tie_break():
    """Identical (state, memory) => identical decision, twice in a row, and
    across two freshly built controllers (no RNG in the decision path)."""
    jc1 = _trained_controller()
    st = _state()
    d1a = jc1.decide(st, now_s=200.0)
    d1b = jc1.decide(st, now_s=200.0)
    assert d1a.action == d1b.action
    assert d1a.state_class == d1b.state_class

    # an independent clone with the same memories decides identically
    jc2 = _trained_controller()
    d2 = jc2.decide(st, now_s=200.0)
    assert d2.action == d1a.action


# ---------------------------------------------------------------------------
# test_graceful_degradation (§52 minimax)
# ---------------------------------------------------------------------------

def test_graceful_degradation():
    """In a bad state the controller still returns an admissible action
    respecting floors; the §52 minimax selector is the argmin of the max
    predicted severity over the floor-respecting set (tested at the
    selector level — deterministic); the decision path degrades without
    claiming a satisfying allocation."""
    jc = _trained_controller(sigma_support=0.999)
    bad = _state(n_e=6, n_u=3, n_m=5, queue_e=1.9e6, queue_u=1e6, queue_m=2e6,
                 kpi_e_raw=1.0, kpi_u_raw=25.0, kpi_m_raw=0.4)
    d = jc.decide(bad, now_s=200.0)
    assert sum(d.action) <= 53
    assert all(a >= f for a, f in zip(d.action, d.floors.floors))

    from nsoran.sqr_jc.action_search import JointActionOracleEvaluator
    from nsoran.sqr_jc.conditional_floors import compute_floors

    ev = JointActionOracleEvaluator(jc.models, sigma_support=jc.sigma_support, c=53)
    res = ev.evaluate(bad.vector(jc.scaler))
    fr = compute_floors(res, (bad.n_e, bad.n_u, bad.n_m),
                        (bad.queue_e, bad.queue_u, bad.queue_m), 53)
    mask = np.all(res.actions >= np.asarray(fr.floors)[None, :], axis=1)
    sev = np.maximum(0.0, -res.safe)
    max_sev = sev.max(axis=1)
    # §52 selector IS the argmin of max severity over the masked set
    sel = jc._select_infeasible(res, mask)
    assert float(max_sev[sel]) == pytest.approx(float(max_sev[mask].min()), rel=1e-12)
    # and the controller's chosen action is that selector's action whenever
    # the state was classified INFEASIBLE
    if d.state_class == "INFEASIBLE":
        assert d.action == tuple(int(v) for v in res.actions[sel])


# ---------------------------------------------------------------------------
# test_shared_bootstrap_determinism (§29/§30)
# ---------------------------------------------------------------------------

def test_shared_bootstrap_determinism():
    """0–100 s: every learning controller emits the SAME PROP allocation;
    decisions are identical across instances and independent of the models."""
    st = _state(window_index=500)  # t = 50 s < 100 s
    from nsoran.sqr.sqr_kpis import SliceKpis, SqrKpmWindow

    window = SqrKpmWindow(timestamp=51000, slices={
        0: SliceKpis(0, 2, 4.0, 3.8, 4.2, 1e5, None, 0, {}, ()),
        1: SliceKpis(1, 1, 1.0, 0.9, 1.0, 100.0, None, 0, {}, ()),
        2: SliceKpis(2, 2, 1.0, 0.9, 1.0, 500.0, None, 0, {}, ()),
    }, total_ues=5, cell_snr_median=20.0, rows=())

    ctl_a = JointController(
        tuple(MultiQuantileKernelModel(dim=DIM) for _ in range(3)),
        _make_scaler(), sigma_support=0.9, c=53)
    ctl_b = JointController(
        tuple(MultiQuantileKernelModel(dim=DIM) for _ in range(3)),
        _make_scaler(), sigma_support=0.5, c=53)
    da = ctl_a.decide(st, now_s=50.0, kpm_window=window)
    db = ctl_b.decide(st, now_s=50.0, kpm_window=window)
    assert da.bootstrap is True and db.bootstrap is True
    assert da.action == db.action
    assert da.action == tuple(int(v) for v in
                              ctl_a.bootstrap_controller.prop_split(window))
    assert sum(da.action) <= 53
    # BOOTSTRAP_SECONDS frozen at 100 s (§29)
    assert BOOTSTRAP_SECONDS == 100.0
    d_learned = ctl_a.decide(st, now_s=100.0001)
    assert d_learned.bootstrap is False


# ---------------------------------------------------------------------------
# test_replay_state_identity (§61 requirement: same state -> same decision)
# ---------------------------------------------------------------------------

def test_replay_state_identity():
    """A controller restored from state_dict replays the identical decision
    for the identical state (replay determinism, protocol §61)."""
    jc = _trained_controller()
    st = _state()
    d_ref = jc.decide(st, now_s=200.0)

    state_payload = {
        "models": [m.state_dict() for m in jc.models],
        "prev_action": jc.prev_action,
        "sigma_support": jc.sigma_support,
        "c": jc.c,
    }
    clone_models = tuple(MultiQuantileKernelModel(dim=DIM, memory_size=100)
                         for _ in range(3))
    for m, payload in zip(clone_models, state_payload["models"]):
        m.load_state_dict(payload)
    # rebuild the frozen scaler from the same manifest state
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scaler.yaml")
        jc.scaler.save_manifest(path)
        scaler2 = FeatureScaler.load_manifest(path)
    jc2 = JointController(clone_models, scaler2,
                          sigma_support=state_payload["sigma_support"], c=53,
                          bootstrap_seconds=0.0)
    jc2.prev_action = state_payload["prev_action"]
    d2 = jc2.decide(st, now_s=200.0)
    assert d2.action == d_ref.action
    assert d2.state_class == d_ref.state_class
    assert d2.floors.floors == d_ref.floors.floors


# ---------------------------------------------------------------------------
# §45: NO post-hoc margin on the joint action
# ---------------------------------------------------------------------------

def test_no_post_hoc_margin_invariant():
    """The selected action equals the evaluated enumeration entry EXACTLY —
    the audited SQR margin a <- a + round(sigma*M) is NOT applied (§45);
    M is maintained only for logs."""
    jc = _trained_controller()
    st = _state()
    d = jc.decide(st, now_s=200.0)
    from nsoran.sqr_jc.joint_action_space import enumerate_actions

    A = enumerate_actions(53)
    hit = (A == np.asarray(d.action)).all(axis=1).sum()
    assert hit == 1  # action is an exact enumeration entry (no +margin edit)
    # M exists (log continuity) but no JointDecision field carries a margin
    assert all(m.margin_m >= 2.0 for m in jc.models)
    assert not hasattr(d, "margin_applied")


# ---------------------------------------------------------------------------
# §56/§57 logging conformance
# ---------------------------------------------------------------------------

def test_logging_exact_columns(tmp_path):
    """The decision CSV carries the §56 columns EXACTLY, in order; rows are
    length-checked; outcome and kernel logs open/close cleanly."""
    assert len(DECISION_COLUMNS) == 46
    assert DECISION_COLUMNS[0] == "time" and DECISION_COLUMNS[-1] == "decision_time_ms"
    for name in ("state_class", "search_mode", "floor_e", "quantile_crossing_m"):
        assert name in DECISION_COLUMNS

    values = [0] * len(DECISION_COLUMNS)
    values[DECISION_COLUMNS.index("state_class")] = "FEASIBLE"
    line = format_row(values)
    assert line.count(",") == len(DECISION_COLUMNS) - 1
    with pytest.raises(ValueError):
        format_row([0] * (len(DECISION_COLUMNS) - 1))

    dpath = os.path.join(str(tmp_path), "decisions.csv")
    opath = os.path.join(str(tmp_path), "outcomes.csv")
    kpath = os.path.join(str(tmp_path), "kernels.csv")
    writer = DecisionLogWriter(dpath, opath, kpath)
    writer.write_decision(values)
    writer.write_outcome([0] * len(OUTCOME_COLUMNS))
    from nsoran.sqr_jc.logging import KernelLogRow

    writer.write_kernel_row(KernelLogRow(
        time_ms=100.0, slice_support_count=(10, 10, 10), insertions=30,
        evictions=0, kernel_evaluations=30, prediction_time_ms=1.0,
        update_time_ms=0.5, out_of_support_count=0,
        tau_safe=(0.05, 0.05, 0.05), margin_m=(10.0, 10.0, 10.0),
    ))
    writer.close()
    header = open(dpath).readline().strip().split(",")
    assert tuple(header) == DECISION_COLUMNS
    assert tuple(open(opath).readline().strip().split(",")) == OUTCOME_COLUMNS
    assert tuple(open(kpath).readline().strip().split(",")) == KERNEL_LOG_COLUMNS


# ---------------------------------------------------------------------------
# SQR-HET (§27): isolation of the delta
# ---------------------------------------------------------------------------

def _make_het():
    scaler = _make_scaler()
    controllers = tuple(
        SqrSliceController(
            state_dim=6, n_prbs=53, gamma=0.0,
            sla_threshold=0.0,          # margin scale: m > 0 = satisfied
            memory_size=64,
        )
        for _ in range(3)
    )
    from nsoran.sqr_jc.sqr_het import SqrHetController

    return SqrHetController(controllers, scaler, c=53)


def test_sqr_het_no_joint_machinery():
    """SQR-HET exposes NO feasibility/floors/joint-state machinery (§27)."""
    het = _make_het()
    from nsoran.sqr_jc.sqr_het import CONTROLLER_NAME

    assert CONTROLLER_NAME == "sqr_het"
    for attr in ("feasibility", "floors", "q25", "q75", "state_class",
                 "joint_action_space"):
        assert not hasattr(het, attr)
        assert not any(hasattr(c, attr) for c in het.controllers)
    # the underlying learner is the AUDITED SqrSliceController
    from nsoran.sqr.sqr_controller import SqrSliceController

    assert all(isinstance(c, SqrSliceController) for c in het.controllers)
    with pytest.raises(TypeError):
        from nsoran.sqr_jc.sqr_het import SqrHetController

        SqrHetController(tuple(object() for _ in range(3)), _make_scaler(), c=53)


def test_sqr_het_slice_wise_plus_projection():
    """Slice-wise decisions + original projection Π; per-slice feature is
    x_i = [z_i, a_i/C] with z_i of dim 6 (NOT the joint state)."""
    het = _make_het()
    st = _state()
    d = het.decide(st)
    assert len(d.proposals) == 3 and len(d.action) == 3
    assert sum(d.action) <= 53
    # proposals can exceed C (slice-wise); Π cuts them back (audited rule)
    z0 = het.slice_state(st, 0)
    assert z0.shape == (6,)
    # online update on the common margin trains the audited regressor
    out = SliceOutcome(embb_throughput_mbps=4.0, urllc_p95_ms=9.0,
                       mmtc_pdr_rolling=0.97)
    recs = het.observe_outcome(st, d.action, out)
    assert len(recs) == 3
    # virtual negative updates exist at cut proposals
    over = _state()  # force an over-capacity proposal via decisions: simulate
    from nsoran.sqr_jc.sqr_het import SqrHetController as _S

    assert hasattr(het, "apply_virtual_updates")


def test_sqr_het_scaler_indexing_audit_emu_bug():
    """Audit (A4 pre-check): document exact FeatureScaler indexing bug in
    ``sqr_het.py`` ('letter = \"emu\"[slice_id]') while slices are (eMBB, URLLC,
    mMTC). Confirms URLLC uses (n_m, queue_m) bounds and mMTC uses (n_u, queue_u)
    bounds, leaving A4 frozen without silently fixing it."""
    from dataclasses import replace
    from nsoran.sqr_jc.joint_state import FEATURE_INDEX
    from nsoran.sqr_jc.sqr_het import SqrHetController

    # Distinct per-slice p99 bounds (p1=0.0, p99=exact cap via 101-row grid).
    caps = {
        "n_e": 10.0, "n_u": 30.0, "n_m": 20.0,
        "queue_e": 100.0, "queue_u": 700.0, "queue_m": 1400.0,
    }
    raw = np.zeros((101, ZDIM), dtype=np.float64)
    for feat, cap in caps.items():
        raw[99:, FEATURE_INDEX[feat]] = cap
    scaler = FeatureScaler(c=53)
    scaler.fit(raw)

    controllers = tuple(
        SqrSliceController(state_dim=6, n_prbs=53, gamma=0.0,
                           sla_threshold=0.0, memory_size=64)
        for _ in range(3)
    )
    het = SqrHetController(controllers, scaler, c=53, mean_field=False)
    st = replace(
        _state(),
        n_e=5, n_u=6, n_m=4,
        queue_e=50.0, queue_u=350.0, queue_m=700.0,
        kpi_e_raw=1.75, kpi_u_raw=5.0, kpi_m_raw=0.90,
        snr_e=15.0, snr_u=15.0, snr_m=15.0,
    )
    z0, z1, z2 = het.slice_state(st, 0), het.slice_state(st, 1), het.slice_state(st, 2)

    # eMBB (slice 0): uses n_e=10, queue_e=100 (unaffected).
    assert np.isclose(z0[0], 5.0 / 10.0)
    assert np.isclose(z0[1], 50.0 / 100.0)

    # URLLC (slice 1): physical values (n_u=6, queue_u=350), scaled by (n_m=20, queue_m=1400)
    # instead of (n_u=30, queue_u=700)!
    assert np.isclose(z1[0], 6.0 / 20.0) and not np.isclose(z1[0], 6.0 / 30.0)
    assert np.isclose(z1[1], 350.0 / 1400.0) and not np.isclose(z1[1], 350.0 / 700.0)

    # mMTC (slice 2): physical values (n_m=4, queue_m=700), scaled by (n_u=30, queue_u=700)
    # instead of (n_m=20, queue_m=1400)!
    assert np.isclose(z2[0], 4.0 / 30.0) and not np.isclose(z2[0], 4.0 / 20.0)
    assert np.isclose(z2[1], 700.0 / 700.0) and not np.isclose(z2[1], 700.0 / 1400.0)

    # KPI & SNR remain correctly aligned per slice.
    assert np.isclose(z1[2], 5.0 / 10.0) and np.isclose(z2[2], 0.90)
    assert np.isclose(z1[3], 15.0 / 30.0) and np.isclose(z2[3], 15.0 / 30.0)

    # Also verify against frozen CAL manifest if present in workspace.
    manifest = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "results", "sqr_jc_final", "03_calibration", "feature_scaler_manifest.yaml",
    ))
    if os.path.exists(manifest):
        cal_scaler = FeatureScaler.load_manifest(manifest)
        het_cal = SqrHetController(controllers, cal_scaler, c=53, mean_field=False)
        st_cal = replace(st, n_u=2, n_m=2, queue_u=709.0, queue_m=709.0)
        zu_cal = het_cal.slice_state(st_cal, 1)
        zm_cal = het_cal.slice_state(st_cal, 2)
        # In CAL manifest: n_u_p99=3, n_m_p99=2; queue_u_p99=709, queue_m_p99=1422.
        assert np.isclose(zu_cal[0], 2.0 / 2.0)       # saturated at n_u=2 instead of 2/3
        assert np.isclose(zu_cal[1], 709.0 / 1422.0)  # half-scaled instead of 1.0
        assert np.isclose(zm_cal[0], 2.0 / 3.0)       # 0.6667 instead of 1.0
        assert np.isclose(zm_cal[1], 709.0 / 709.0)   # saturated at 709 B instead of ~0.4986


def test_sqr_het_mean_field_a4_isolation_and_causality():
    """A4 (--mean_field): replaces dims 4..5 with [prb_others_norm, total_ues_norm]
    from pre-decision state while keeping dims 0..3 bit-identical and R11 intact."""
    from dataclasses import replace
    from nsoran.sqr.sqr_env import N_MAX_CELL, clip_norm
    from nsoran.sqr_jc.sqr_het import SqrHetController

    scaler = _make_scaler()
    ctls_off = tuple(
        SqrSliceController(state_dim=6, n_prbs=53, gamma=0.0,
                           sla_threshold=0.0, memory_size=64)
        for _ in range(3)
    )
    ctls_on = tuple(
        SqrSliceController(state_dim=6, n_prbs=53, gamma=0.0,
                           sla_threshold=0.0, memory_size=64)
        for _ in range(3)
    )
    het_off = SqrHetController(ctls_off, scaler, c=53, mean_field=False)
    het_on = SqrHetController(ctls_on, scaler, c=53, mean_field=True)

    st = replace(
        _state(),
        n_e=3, n_u=2, n_m=5,
        prev_a_e=18, prev_a_u=12, prev_a_m=15,
        viol_e=1, viol_u=0, viol_m=1,
    )
    expected_total_ues = clip_norm(3 + 2 + 5, N_MAX_CELL)  # 10 / 20 = 0.5
    expected_others = (
        clip_norm(12 + 15, 53),  # slice 0
        clip_norm(18 + 15, 53),  # slice 1
        clip_norm(18 + 12, 53),  # slice 2
    )
    for sid in range(3):
        z_off = het_off.slice_state(st, sid)
        z_on = het_on.slice_state(st, sid)
        # Dims 0..3 are bit-identical (preserves scaler semantics and bug isolation).
        assert np.array_equal(z_on[:4], z_off[:4])
        # A0 keeps [prev_a_i/C, viol_i].
        prev_i = (18, 12, 15)[sid]
        viol_i = (1, 0, 1)[sid]
        assert np.isclose(z_off[4], prev_i / 53.0) and np.isclose(z_off[5], float(viol_i))
        # A4 restores [prb_others_norm, total_ues_norm].
        assert np.isclose(z_on[4], expected_others[sid])
        assert np.isclose(z_on[5], expected_total_ues)

    # Causality at t=0 (prev_a = (0,0,0)) and empty slice (n_u=0):
    st_t0 = replace(_state(), n_e=4, n_u=0, n_m=2, prev_a_e=0, prev_a_u=0, prev_a_m=0)
    for sid in range(3):
        z0 = het_on.slice_state(st_t0, sid)
        assert z0[4] == 0.0
        assert np.isclose(z0[5], clip_norm(6, N_MAX_CELL))

    # R11 margin adaptation responds identically to violated outcome (m < 0)
    # without depending on z_i[4..5] (adapts at violation_window=100).
    out_viol = SliceOutcome(embb_throughput_mbps=0.5, urllc_p95_ms=25.0,
                            mmtc_pdr_rolling=0.50)
    m_before = [c.margin for c in het_on.controllers]
    tau_before = [c.tau for c in het_on.controllers]
    for _ in range(het_on.controllers[0].violation_window):
        recs_off = het_off.observe_outcome(st, (18, 12, 15), out_viol)
        recs_on = het_on.observe_outcome(st, (18, 12, 15), out_viol)
        assert len(recs_on) == 3 and len(recs_off) == 3
    assert [c.tau for c in het_on.controllers] == [c.tau for c in het_off.controllers]
    assert [c.margin for c in het_on.controllers] == [c.margin for c in het_off.controllers]
    assert all(c.margin > mb for c, mb in zip(het_on.controllers, m_before))
    assert all(c.tau < tb for c, tb in zip(het_on.controllers, tau_before))


def test_sqr_het_scaler_indexing_fix_eum_a_scaler():
    """A_scaler (--fix_scaler_index): maps slice_id -> 'eum'[slice_id] so
    eMBB, URLLC, and mMTC each use their own user-count and queue scalers
    while keeping dims 2..5 bit-identical and preserving fix_scaler_index=False."""
    from dataclasses import replace
    from nsoran.sqr_jc.joint_state import FEATURE_INDEX
    from nsoran.sqr_jc.sqr_het import SqrHetController

    caps = {
        "n_e": 10.0, "n_u": 30.0, "n_m": 20.0,
        "queue_e": 100.0, "queue_u": 700.0, "queue_m": 1400.0,
    }
    raw = np.zeros((101, ZDIM), dtype=np.float64)
    for feat, cap in caps.items():
        raw[99:, FEATURE_INDEX[feat]] = cap
    scaler = FeatureScaler(c=53)
    scaler.fit(raw)

    ctls_bug = tuple(
        SqrSliceController(state_dim=6, n_prbs=53, gamma=0.0,
                           sla_threshold=0.0, memory_size=64)
        for _ in range(3)
    )
    ctls_fix = tuple(
        SqrSliceController(state_dim=6, n_prbs=53, gamma=0.0,
                           sla_threshold=0.0, memory_size=64)
        for _ in range(3)
    )
    het_bug = SqrHetController(ctls_bug, scaler, c=53, fix_scaler_index=False)
    het_fix = SqrHetController(ctls_fix, scaler, c=53, fix_scaler_index=True)

    st = replace(
        _state(),
        n_e=5, n_u=6, n_m=4,
        queue_e=50.0, queue_u=350.0, queue_m=700.0,
        kpi_e_raw=1.75, kpi_u_raw=5.0, kpi_m_raw=0.90,
        snr_e=15.0, snr_u=15.0, snr_m=15.0,
    )
    for sid in range(3):
        zb = het_bug.slice_state(st, sid)
        zf = het_fix.slice_state(st, sid)
        assert np.array_equal(zf[2:], zb[2:])

    z0 = het_fix.slice_state(st, 0)
    z1 = het_fix.slice_state(st, 1)
    z2 = het_fix.slice_state(st, 2)
    assert np.isclose(z0[0], 5.0 / 10.0) and np.isclose(z0[1], 50.0 / 100.0)
    assert np.isclose(z1[0], 6.0 / 30.0) and np.isclose(z1[1], 350.0 / 700.0)
    assert np.isclose(z2[0], 4.0 / 20.0) and np.isclose(z2[1], 700.0 / 1400.0)

    manifest = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "results", "sqr_jc_final", "03_calibration", "feature_scaler_manifest.yaml",
    ))
    if os.path.exists(manifest):
        cal_scaler = FeatureScaler.load_manifest(manifest)
        het_cal = SqrHetController(ctls_fix, cal_scaler, c=53, fix_scaler_index=True)
        st_cal = replace(st, n_e=2, n_u=2, n_m=2, queue_e=1000.0, queue_u=709.0, queue_m=709.0)
        zu_cal = het_cal.slice_state(st_cal, 1)
        zm_cal = het_cal.slice_state(st_cal, 2)
        assert np.isclose(zu_cal[0], 2.0 / 3.0)
        assert np.isclose(zu_cal[1], 709.0 / 709.0)
        assert np.isclose(zm_cal[0], 2.0 / 2.0)
        assert np.isclose(zm_cal[1], 709.0 / 1422.0)

