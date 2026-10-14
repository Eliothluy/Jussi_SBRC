"""Unit tests for the SQR xApp E2 bridge (Fase 5.4).

Covers the RAN-slicing integer-percent machinery that guarantees IPC x E2
parity (``examples/xapp_sqr.py``):

- ``compute_rbg_quotas_py`` — exact replica of
  ``SqrMacScheduler::ComputeRbgQuotas`` (scratch/sqr/sqr-mac-scheduler.cc).
  The accumulation is a NAIVE left fold on purpose: Python's ``sum()`` is
  compensated (Neumaier) since 3.12 and diverges from the C++ in the last
  ulp, which flips the ``llround`` boundary (real case found in the Fase 5.4
  parity: pcts (2,12,36) @ totalRbg=53 -> naive 26.5 -> target 27 vs
  compensated 26.499999999999996 -> target 26).
- ``integer_pcts_for_quotas`` — integer percent vector (the RIC Control
  wire format, ASN.1 INTEGER) reproducing the float-percent RBG quotas
  EXACTLY.
- The indication -> contract §4 KPM materialization round-trips through
  ``SqrKpmParser`` with the same derivations as the file-based path.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xapp_sqr import (  # noqa: E402
    compute_rbg_quotas_py,
    integer_pcts_for_quotas,
    window_info,
    write_kpm_window,
)

from nsoran.sqr.sqr_kpis import SqrKpmParser  # noqa: E402

RBG = 53   # primary scenario grid (register R01: total_prb = totalRbg = 53)
C = 53
L = 3


def test_replica_documented_cpp_examples():
    """The two worked examples of the Fase 5.3 report (fase5.3 §3.1)."""
    assert compute_rbg_quotas_py([34.0, 33.0, 33.0], RBG) == [18, 18, 17]
    assert compute_rbg_quotas_py([17.0, 17.0, 16.0], RBG) == [9, 9, 9]


def test_replica_naive_fold_not_compensated_sum():
    """The llround boundary case: the naive fold must give target 27."""
    pcts = [2.0, 12.0, 36.0]
    assert compute_rbg_quotas_py(pcts, RBG) == [1, 7, 19]
    # Prove the C++ semantics: naive accumulation of the raw values.
    raw = [p / 100.0 * float(RBG) for p in pcts]
    acc = 0.0
    for r in raw:
        acc += r
    assert acc == 26.5  # compensated sum() would give 26.499999999999996
    assert math.floor(acc + 0.5) == 27


def test_replica_zero_and_full_partition():
    assert compute_rbg_quotas_py([0.0, 0.0, 0.0], RBG) == [0, 0, 0]
    assert compute_rbg_quotas_py([100.0, 0.0, 0.0], RBG) == [RBG, 0, 0]
    # Σ pct = 100 exactly: full partition by largest remainders, tie-break to
    # the lowest slice index (the SQR agent guarantees Σa <= C by projection;
    # Σ pct > 100 is OUTSIDE the contract — the C++ NS_ASSERTs there).
    q = compute_rbg_quotas_py([33.4, 33.3, 33.3], RBG)
    assert sum(q) == RBG and q == [18, 18, 17]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_replica_guarantees_random_vectors(seed):
    """Contract domain (Σ pct <= 100 — the projection guarantees Σa <= C):
    0 <= Σ quota <= totalRbg and floor-consistency on random vectors."""
    rng = np.random.default_rng(seed)
    for _ in range(500):
        total = rng.uniform(0.0, 100.0)
        splits = np.sort(rng.uniform(0.0, total, size=L - 1))
        pct = np.concatenate([[splits[0]], np.diff(splits), [total - splits[-1]]])
        q = compute_rbg_quotas_py(pct.tolist(), RBG)
        assert len(q) == L
        assert all(0 <= v for v in q)
        assert sum(q) <= RBG
        raw = [min(max(p, 0.0), 100.0) / 100.0 * RBG for p in pct]
        for qi, ri in zip(q, raw):
            assert qi == int(math.floor(ri)) or qi == int(math.floor(ri)) + 1


def test_search_reproduces_quotas_full_small_space():
    """Integer pcts reproduce the float quotas exactly over a full slice-1 sweep."""
    for a in range(C + 1):
        pctf = [100.0 * a / C, 25.0, 25.0]
        pcts, target, ok = integer_pcts_for_quotas(pctf, RBG)
        assert ok, f"a={a}: no exact integer pcts"
        assert compute_rbg_quotas_py(pcts, RBG) == target


def test_search_sampled_grid_exact():
    rng = np.random.default_rng(7)
    for _ in range(300):
        acts = rng.integers(0, C + 1, size=L)
        total = int(acts.sum())
        if total > C:  # project (Pi, R24) to keep Σa <= C
            excess = total - C
            for i in range(L - 1, -1, -1):
                take = min(int(acts[i]), excess)
                acts[i] -= take
                excess -= take
                if excess == 0:
                    break
        pctf = [100.0 * int(a) / C for a in acts]
        pcts, target, ok = integer_pcts_for_quotas(pctf, RBG)
        assert ok
        assert compute_rbg_quotas_py(pcts, RBG) == target
        assert all(0 <= p <= 100 for p in pcts)


def test_write_kpm_window_parses_like_the_file_path(tmp_path):
    """Indication -> materialized KPM -> SqrKpmParser yields the §4 derivations."""
    ind = {
        "timestamp_ms": 300,
        "ue_records": [
            # imsi, rnti, slice, dtx, drx, plr, rsh, dlost, thr, buf, wthr, snr, queue
            [1, 11, 0, 1000, 900, 0.5, 33.33, 0, 0.72, 2000, 0.72, 12.34, 1500],
            [2, 12, 1, 2000, 1800, 0.25, 20.0, 1, 1.51, 1000, 0.95, -999.0, 900],
        ],
    }
    path = str(tmp_path / "e2_kpm_indication.csv")
    assert write_kpm_window(path, ind) == 2
    w = SqrKpmParser(sla_thresholds={0: 0.5, 1: 1.0, 2: 1.5}).parse_window(path)
    assert w.timestamp == 300 and w.total_ues == 2
    assert w.slices[0].n_ues == 1
    assert w.slices[0].p5_thr_mbps == pytest.approx(0.72)
    # slice 0 SLA is 0.5 and thr 0.72 >= 0.5 -> violation_rate 0.0
    assert w.slices[0].violation_rate == pytest.approx(0.0)
    # slice 1 SNR sentinel must be imputed (never -999 in the state).
    assert w.slices[1].snr_values[0] != -999.0
    assert w.slices[1].n_snr_imputed == 1
    info = window_info(w)
    assert info["kpm_timestamp"] == 300
    assert info["reward"] == pytest.approx(-(1 / 2))  # 1 of 2 UE-windows violating
