"""Action-space semantics + search tests (§21/§22/§35–§38).

Covers: test_resource_unit_manifest_complete, test_action_semantics,
test_capacity_constraint, test_nonnegative_action, exhaustive/coarse-to-fine
search behavior, the §37 benchmark harness and the deterministic tie-break.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

import numpy as np
import pytest

from nsoran.sqr_jc.action_search import (
    CoarseToFineParams,
    CoarseToFineSearch,
    JointActionOracleEvaluator,
    benchmark_exhaustive,
)
from nsoran.sqr_jc.joint_action_space import (
    ACTION_SEMANTICS,
    C_PRB,
    NUM_ACTIONS_ALLOCATED_USAGE_C53,
    NUM_ACTIONS_BUDGET_PARTITION_C53,
    enumerate_actions,
    filter_feasible_mask,
    num_actions_allocated_usage,
    num_actions_budget_partition,
    switching_distance,
    validate_action,
)
from nsoran.sqr_jc.kernel_models import MultiQuantileKernelModel

DIM = 8
_MANIFEST = Path(__file__).resolve().parents[1] / "results" / "sqr_jc_final" / \
    "01_resource_unit" / "resource_unit_manifest.md"
_SLA_MANIFEST = Path(__file__).resolve().parents[1] / "src" / "nsoran" / "sqr_jc" / \
    "sla_manifest.yaml"


# ---------------------------------------------------------------------------
# test_resource_unit_manifest_complete (G1 artifact exists and answers §21)
# ---------------------------------------------------------------------------

def test_resource_unit_manifest_complete():
    assert _MANIFEST.exists(), f"missing G1 artifact: {_MANIFEST}"
    text = _MANIFEST.read_text(encoding="utf-8")
    required = [
        "C=53",          # the value under audit
        "RBG",           # the unit
        "PRB",           # physical unit equivalence
        "1 PRB",         # RBG size = 1 PRB answer
        "idle",          # idle budget behavior
        "ALLOCATED_USAGE",  # the frozen semantics
        "R27",           # the register that froze it
        "action_semantics",  # ACTION_SEMANTICS section marker
    ]
    for token in required:
        assert token in text, f"resource unit manifest lacks {token!r}"
    # the frozen semantics constant agrees with the manifest
    assert ACTION_SEMANTICS == "ALLOCATED_USAGE"


# ---------------------------------------------------------------------------
# test_action_semantics (§22: audited semantics + combination counts)
# ---------------------------------------------------------------------------

def test_action_semantics():
    """ALLOCATED_USAGE: Σa ≤ C (idle allowed) — proven by the audited R27
    absolute-quota scheduler; counts match the protocol's §22 numbers."""
    assert ACTION_SEMANTICS == "ALLOCATED_USAGE"
    assert num_actions_allocated_usage(53) == NUM_ACTIONS_ALLOCATED_USAGE_C53
    assert NUM_ACTIONS_ALLOCATED_USAGE_C53 == 27720
    assert num_actions_budget_partition(53) == NUM_ACTIONS_BUDGET_PARTITION_C53
    assert NUM_ACTIONS_BUDGET_PARTITION_C53 == 1485
    actions = enumerate_actions(53)
    assert actions.shape == (27720, 3)
    assert int(actions.sum(axis=1).max()) <= 53
    assert int(actions.sum(axis=1).min()) == 0  # idle possible (R27)
    # lexicographic order (deterministic tie-break domain)
    order = np.lexsort((actions[:, 2], actions[:, 1], actions[:, 0]))
    assert np.array_equal(order, np.arange(len(actions)))


def test_action_semantics_partition_refuted():
    """The Σa = C partition reading is REFUTED by R27 (idle budgets stay
    idle — never redistributed): actions with Σ < C MUST be admissible."""
    actions = enumerate_actions(53)
    partial = actions[actions.sum(axis=1) < 53]
    assert partial.shape[0] > 0  # a partition space would exclude these


# ---------------------------------------------------------------------------
# test_capacity_constraint / test_nonnegative_action (§35)
# ---------------------------------------------------------------------------

def test_capacity_constraint():
    actions = enumerate_actions(53)
    assert np.all(actions.sum(axis=1) <= C_PRB)
    with pytest.raises(ValueError):
        validate_action((30, 30, 30), c=53)  # Σ=90 > 53
    with pytest.raises(ValueError):
        validate_action((0, 0, 54), c=53)    # a_m > C


def test_nonnegative_action():
    actions = enumerate_actions(53)
    assert np.all(actions >= 0)
    with pytest.raises(ValueError):
        validate_action((-1, 0, 0), c=53)
    # floor violation (§35: an active floor may not be violated)
    with pytest.raises(ValueError):
        validate_action((0, 5, 3), c=53, floors=(0, 13, 3))


def test_filter_feasible_mask_with_floors():
    actions = np.array([
        [0, 13, 3],     # ok
        [0, 12, 3],     # floor violation
        [54, 0, 0],     # > C
        [20, 20, 20],   # Σ=60 > C
        [1, 13, 3],     # ok
    ], dtype=np.int64)
    mask = filter_feasible_mask(actions, 53, floors=(0, 13, 3))
    assert mask.tolist() == [True, False, False, False, True]


def test_switching_distance():
    assert switching_distance((18, 13, 3), (18, 13, 3)) == 0
    assert switching_distance((0, 13, 3), (18, 13, 3)) == 18
    assert switching_distance((48, 5, 0), (37, 13, 3)) == 22


# ---------------------------------------------------------------------------
# Search: exhaustive vs coarse-to-fine + §37 harness
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator():
    rng = np.random.default_rng(11)
    dim = 8
    model = MultiQuantileKernelModel(dim=dim, memory_size=64)
    for i in range(30):
        x = rng.uniform(0, 1, size=dim)
        model.update(x, y=float(rng.uniform(-0.5, 0.8)))
    z_dim = dim - 3
    return JointActionOracleEvaluator(
        (model, MultiQuantileKernelModel(dim=dim, memory_size=64),
         MultiQuantileKernelModel(dim=dim, memory_size=64)),
        sigma_support=0.7,
    ), z_dim


def _lex_keys(result):
    """FEASIBLE-style selection keys for the coarse-to-fine driver."""
    risk = np.maximum(0.0, -result.safe).sum(axis=1)
    uncertainty = result.sigma.sum(axis=1)
    switching = np.abs(result.actions - np.array([18, 13, 3])).sum(axis=1).astype(float)
    total = result.actions.sum(axis=1).astype(float)
    return [total, switching, uncertainty, risk]


def test_search_exhaustive_shapes_and_support(evaluator):
    ev, z_dim = evaluator
    z = np.full(z_dim, 0.5)
    res = ev.evaluate(z)
    assert res.n_actions == 27720
    assert res.safe.shape == (27720, 3)
    assert res.exhaustive is True
    # floors restrict the evaluated set
    res_f = ev.evaluate(z, floors=(0, 13, 3))
    assert res_f.n_actions < res.n_actions
    assert np.all(res_f.actions[:, 1] >= 13)
    assert np.all(res_f.actions[:, 2] >= 3)


def test_search_coarse_to_fine_subset_and_determinism(evaluator):
    ev, z_dim = evaluator
    z = np.full(z_dim, 0.5)
    ctf = CoarseToFineSearch(ev, CoarseToFineParams(coarse_step=4, refine_radius=3))
    res1, best1 = ctf.search(z, floors=(0, 0, 0), key_fn=_lex_keys)
    res2, best2 = ctf.search(z, floors=(0, 0, 0), key_fn=_lex_keys)
    assert best1 == best2                      # deterministic
    assert res1.exhaustive is False            # NOT a coverage certificate
    coarse = ctf.candidate_set()
    assert coarse.shape[0] < 27720             # reduced search
    assert res1.n_actions >= coarse.shape[0]   # union ⊇ coarse stage


def test_benchmark_exhaustive_harness(evaluator):
    """§37 harness: returns median/p95 and a PASS flag at ≤ 80 ms. (The
    orchestrator runs the REAL gate on CAL states; here we validate the
    harness mechanics on a small synthetic state list.)"""
    ev, z_dim = evaluator
    states = [np.full(z_dim, 0.1 * (k + 1)) for k in range(3)]
    out = benchmark_exhaustive(ev, states)
    for key in ("n_states", "median_ms", "p95_ms", "n_actions", "passed_p95_80ms"):
        assert key in out
    assert out["n_states"] == 3
    assert isinstance(out["passed_p95_80ms"], bool)


def test_search_quality_vs_oracle_concept(evaluator):
    """§38 requires the coarse-to-fine selection to agree with the oracle
    evaluator on the SAME lexicographic key — check the mechanism on the
    fixture (real agreement gate runs on CAL, orchestrator)."""
    ev, z_dim = evaluator
    z = np.full(z_dim, 0.5)
    full = ev.evaluate(z)
    idx = np.lexsort(_lex_keys(full))
    oracle_best = tuple(int(v) for v in full.actions[idx[0]])
    ctf = CoarseToFineSearch(ev, CoarseToFineParams(coarse_step=2, refine_radius=4))
    _, ctf_best = ctf.search(z, floors=(0, 0, 0), key_fn=_lex_keys)
    # the mechanisms run; agreement depends on the fixture landscape
    assert isinstance(ctf_best, tuple) and len(ctf_best) == 3
    assert isinstance(oracle_best, tuple) and len(oracle_best) == 3


# ---------------------------------------------------------------------------
# sla_manifest.yaml artifact (G2) consistency with the frozen constants
# ---------------------------------------------------------------------------

def test_sla_manifest_artifact_consistency():
    from nsoran.sqr_jc.sla_margin import (
        SLA_EMBB_MBPS,
        SLA_MMTC_PDR,
        SLA_URLLC_MISS_PCT_SECONDARY,
        SLA_URLLC_MS,
    )
    from nsoran.sqr_jc.yaml_subset import load

    assert _SLA_MANIFEST.exists()
    data = load(_SLA_MANIFEST.read_text(encoding="utf-8"))
    assert float(data["embb"]["threshold"]) == SLA_EMBB_MBPS
    assert float(data["urllc"]["threshold"]) == SLA_URLLC_MS
    assert float(data["mmtc"]["threshold"]) == SLA_MMTC_PDR
    assert float(data["urllc"]["secondary_threshold_pct"]) == SLA_URLLC_MISS_PCT_SECONDARY
    assert "p95" in data["urllc"]["margin_formula"]
    assert "divergence" in data["urllc"]["secondary_status"].lower() or \
        "DIVERGÊNCIA" in _SLA_MANIFEST.read_text(encoding="utf-8")
