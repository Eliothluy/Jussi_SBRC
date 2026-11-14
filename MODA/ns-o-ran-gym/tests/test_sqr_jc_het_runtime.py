"""Regression tests for the SQR-HET runtime glue (D1).

Incident D1-v1 (09/11): positional parsing of sqr_packet_delays.csv read
``ueImsi`` as ``txSec`` (wire order is ueImsi,sliceId,txSec,delaySec) and
the URLLC stream came out EMPTY — 8 runs quarentined.  These tests pin:
(i) name-based parsing under the REAL and under permuted headers; (ii) the
frozen percentile/miss semantics; (iii) the mMTC rolling PDR window.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.sqr_jc.het_runtime import DelayStream  # noqa: E402

REAL_HEADER = "ueImsi,sliceId,txSec,delaySec\n"
ALT_HEADER = "txSec,delaySec,sliceId,ueImsi\n"   # ordem distinta — deve dar no mesmo


def _write(path, header, rows):
    with open(path, "w") as f:
        f.write(header)
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def test_parses_real_header_order():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sqr_packet_delays.csv")
    _write(p, REAL_HEADER, [
        (10, 1, 1.001, 0.005),   # URLLC win 10
        (11, 1, 1.002, 0.012),   # URLLC win 10 (miss)
        (99, 0, 1.003, 0.009),   # eMBB — ignored
    ])
    ds = DelayStream(p)
    assert ds.poll() == 2
    s = ds.stats(10)
    assert s["n_pkts"] == 2
    assert abs(s["miss_rate"] - 0.5) < 1e-9
    assert abs(s["p95_ms"] - 11.65) < 1e-6   # interp linear 0.95*(n-1) (frozen)


def test_parses_permuted_header_order():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sqr_packet_delays.csv")
    _write(p, ALT_HEADER, [
        (1.001, 0.005, 1, 10),
        (1.002, 0.012, 1, 11),
    ])
    ds = DelayStream(p)
    assert ds.poll() == 2
    s = ds.stats(10)
    assert s["n_pkts"] == 2 and s["miss_rate"] == 0.5


def test_incremental_poll_and_missing_file():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sqr_packet_delays.csv")
    ds = DelayStream(p)
    assert ds.poll() == 0                      # arquivo ainda não existe
    _write(p, REAL_HEADER, [(10, 1, 1.001, 0.005)])
    assert ds.poll() == 1
    with open(p, "a") as f:
        f.write("11,1,1.101,0.004\n")
    assert ds.poll() == 1                      # incremental
    assert ds.stats(11)["n_pkts"] == 1


def test_urlcc_stream_nonempty_is_the_d1_invariant():
    """The D1 invariant: with URLLC rows on the wire, stats MUST appear."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sqr_packet_delays.csv")
    _write(p, REAL_HEADER, [(10, 1, 1.0, 0.006)] * 50)
    ds = DelayStream(p)
    assert ds.poll() == 50
    s = ds.stats(10)
    assert s["n_pkts"] == 50 and s["miss_rate"] == 0.0


def test_resolves_file_two_levels_below_sim_path(tmp_path):
    """Incident D1-v2: the stream lives at
    <sim>/results_sqr_network_only/seed=*/sqr_packet_delays.csv — the
    runtime must find it via GLOB, not at <sim>/sqr_packet_delays.csv."""
    from nsoran.sqr_jc.het_runtime import HetRuntime
    sim = tmp_path / "uuid-1234"
    tel = sim / "results_sqr_network_only" / "seed=1_run=1"
    tel.mkdir(parents=True)
    _write(str(tel / "sqr_packet_delays.csv"), REAL_HEADER,
           [(10, 1, 1.001, 0.005), (10, 1, 1.002, 0.012)])
    hr = HetRuntime(str(sim))
    assert hr.delays.poll() == 2
    assert hr.delays.stats(10)["n_pkts"] == 2


def test_seq_replay_off_by_one(tmp_path):
    """act() call k aplica a ação da DECISÃO k (seq[k-1]) — nunca k+1
    (incidente A/A D2: janela k com a ação de k+1 divergia o run todo)."""
    import numpy as np
    from nsoran.sqr.replay_controller import ReplayController
    seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    ctl = ReplayController(slice_id=0, n_prbs=53, base="seq",
                           replay_at_win=3, replay_action=(0, 0, 0),
                           seq_actions=seq)
    acts = [ctl.act(np.zeros(6)).action for _ in range(3)]
    assert acts == [1, 4, 0]   # janela1=seq[0][0], janela2=seq[1][0], t=override


# ---------------------------------------------------------------------------
# Delayed feedback URLLC (Emenda candidata — teste isolado 12/11): pool-closure
# ONLINE testável (delivered==tx, ambos disponíveis no instante).  Pin: OPEN
# antes de todas as entregas, CLOSED com stats COMPLETOS depois, NO_DEMAND
# com tx=0 (R19 vacuous), e urllc_only_outcome com margens só-URLLC.
# ---------------------------------------------------------------------------
def test_urllc_pool_status_lifecycle(tmp_path):
    from nsoran.sqr_jc.het_runtime import HetRuntime

    p = tmp_path / "sqr_packet_delays.csv"
    _write(str(p), REAL_HEADER, [
        (10, 1, 1.001, 0.005),   # URLLC win 10 — entrega 1
    ])
    rt = HetRuntime(str(tmp_path))
    rt.delays.path = str(p)
    rt.delays._fh = None
    # janela 10 com tx=2 (KPM) — 1 entrega ainda não flushada
    rt._win_cache[10] = {"embb_median": None, "mmtc_pdr": None,
                         "mmtc_demand": False, "urllc_tx": 2}
    assert rt.urllc_pool_status(10)[0] == "OPEN"
    # tx=0 ⇒ vacuous (R19)
    rt._win_cache[11] = {"urllc_tx": 0}
    assert rt.urllc_pool_status(11)[0] == "NO_DEMAND"
    # segunda entrega chega ao stream ⇒ pool fecha ⇒ stats completos
    _write(str(p), REAL_HEADER, [
        (10, 1, 1.001, 0.005),
        (11, 1, 1.002, 0.012),   # win 10, miss
    ])
    tag, st = rt.urllc_pool_status(10)
    assert tag == "CLOSED" and st["n_pkts"] == 2 and st["miss_rate"] == 0.5
    oc = rt.urllc_only_outcome(10)
    assert oc is not None and oc.urllc_p95_ms is not None
    assert oc.embb_throughput_mbps is None and oc.mmtc_pdr_rolling is None
