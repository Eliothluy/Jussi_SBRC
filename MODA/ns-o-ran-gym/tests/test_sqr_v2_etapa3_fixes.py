"""Regression & sanity tests for Etapa 3A fixes (BUG-01, BUG-06, BUG-07, BUG-08, Patch 1)."""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(GYM_ROOT, "..", ".."))
SQR_V2_CPP_DIR = os.path.join(REPO_ROOT, "MODA", "ns-3-dev", "scratch", "sqr_v2")
EXAMPLES_DIR = os.path.join(GYM_ROOT, "examples")
SRC_DIR = os.path.join(GYM_ROOT, "src")

for p in (SRC_DIR, EXAMPLES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from nsoran.sqr.sqr_kpis import IMP_CARRIED, IMP_CELL_MEDIAN, SQR_KPM_COLUMNS, SQR_KPM_COLUMNS_V13
from nsoran.sqr_v2.campaign_analysis_v2 import compute_seed_scalars_v2, load_prb_windows_df
from nsoran.sqr_v2.env_v2 import (
    IMP_OUT_OF_RANGE_FLOOR,
    SqrV2KpmParser,
    SqrV2ObservationBuilder,
)
import sqr_v2_run

SLAS = {0: 0.5, 1: 1.0, 2: 1.5}
V13_HEADER = SQR_KPM_COLUMNS + SQR_KPM_COLUMNS_V13


def _write_v13_kpm(path, rows):
    lines = [",".join(V13_HEADER)]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_bug01_forced_mcs15_override_absent_and_patch1_present():
    """BUG-01 override removed in sqr_v2 and Patch 1 defensive PDCP guard restored."""
    for fname in os.listdir(SQR_V2_CPP_DIR):
        if fname.endswith((".cc", ".h")):
            text = open(os.path.join(SQR_V2_CPP_DIR, fname), "r", encoding="utf-8").read()
            # Match executable statement `ueInfo->m_dlMcs = 15;` (comments allowed)
            non_comment_lines = [
                ln for ln in text.splitlines() if not ln.lstrip().startswith(("//", "*"))
            ]
            joined = "\n".join(non_comment_lines)
            assert not re.search(r"m_dlMcs\s*=\s*15\s*;", joined), (
                f"BUG-01 forced MCS=15 still active in {fname}"
            )

    pdcp_cc = os.path.join(REPO_ROOT, "MODA", "ns-3-dev", "contrib", "nr", "model", "nr-pdcp.cc")
    pdcp_src = open(pdcp_cc, "r", encoding="utf-8").read()
    assert "p->GetSize() < 2" in pdcp_src
    assert "(firstByte & 0x80) == 0" in pdcp_src


def test_bug06_v2_kpm_parser_out_of_range_and_missing_cqi(tmp_path):
    """BUG-06: wbCqi=0 -> 0.0 dB outage floor (not -999 or 15 dB); wbCqi=-1 uses carried or 0.0 dB."""
    kpm1 = _write_v13_kpm(
        tmp_path / "kpm1.txt",
        [
            [100, 1, 0, 1000, 900, 0.0, 33.3, 0, 0.9, 0, 0.9, "nan", 0, 10, 9, 0.01, 0],
            [100, 2, 1, 1000, 900, 0.0, 33.3, 0, 1.2, 0, 1.2, 21.0, 0, 10, 9, 0.01, 12],
            [100, 3, 2, 1000, 900, 0.0, 33.3, 0, 0.0, 0, 0.0, "nan", 0, 10, 0, 0.0, -1],
        ],
    )
    builder = SqrV2ObservationBuilder(num_slices=3, total_prbs=53, sla_thresholds=SLAS)
    w1 = builder.update_from_kpm(kpm1)

    # Slice 0 (wbCqi=0 explicit outage): 0.0 dB exact, not imputed to 15 dB
    assert w1.slices[0].snr_values == (0.0,)
    assert w1.slices[0].n_snr_imputed == 0
    # Slice 1 (valid CQI=12, 21.0 dB): passes through
    assert w1.slices[1].snr_values == (21.0,)
    assert w1.slices[1].n_snr_imputed == 0
    # Slice 2 (wbCqi=-1, no prior): floor 0.0 dB ("out_of_range_floor"), never -999 or 15 dB
    assert w1.slices[2].snr_values == (0.0,)
    assert w1.slices[2].n_snr_imputed == 1
    assert w1.slices[2].snr_imputations == {IMP_OUT_OF_RANGE_FLOOR: 1}

    obs1 = builder.build_obs()
    assert obs1[0, 3] == pytest.approx(0.0)
    assert obs1[1, 3] == pytest.approx(21.0 / 30.0)
    assert obs1[2, 3] == pytest.approx(0.0)
    assert np.all(np.isfinite(obs1))

    # Next window: UE 2 loses CQI report (-1) -> carried 21.0 dB; UE 3 (-1) -> previous valid cell median
    expected_prev_median = float(np.median([0.0, 21.0]))
    assert w1.cell_snr_median == pytest.approx(expected_prev_median)
    kpm2 = _write_v13_kpm(
        tmp_path / "kpm2.txt",
        [
            [200, 2, 1, 1000, 900, 0.0, 33.3, 0, 1.0, 0, 1.0, "nan", 0, 10, 9, 0.01, -1],
            [200, 3, 2, 1000, 900, 0.0, 33.3, 0, 0.5, 0, 0.5, "nan", 0, 10, 5, 0.01, -1],
        ],
    )
    w2 = builder.update_from_kpm(kpm2)
    assert w2.slices[1].snr_values == (21.0,)
    assert w2.slices[1].snr_imputations == {IMP_CARRIED: 1}
    assert w2.slices[2].snr_values == (pytest.approx(expected_prev_median),)
    assert w2.slices[2].snr_imputations == {IMP_CELL_MEDIAN: 1}


def test_etapa3a_real_ns3_sanity_conservation_and_no_ghosts_or_999(tmp_path):
    """End-to-end 300 m real ns-3 sanity run verifying BUG-06, BUG-07, and BUG-08 fixes."""
    out_dir = tmp_path / "etapa3a_smoke"
    rc = sqr_v2_run.main(
        [
            "--controller", "sqr",
            "--gamma", "1",
            "--seed", "104",
            "--simTime", "8.0",
            "--cellRadius", "300",
            "--sliceLambdas", "24,24,24",
            "--sliceApps", "greedy,urllc,mmtc",
            "--warmup_s", "1.0",
            "--ipc-mode", "strict_rt",
            "--ipc-timeout-ms", "50",
            "--checkpoint_interval", "0",
            "--output", str(out_dir),
            "--run-id", "etapa3a_sanity_s104",
        ]
    )
    assert rc == 0

    df_step = pd.read_csv(out_dir / "step_metrics.csv")
    prb_win = load_prb_windows_df(str(out_dir))
    assert prb_win is not None and not prb_win.empty

    scalars = compute_seed_scalars_v2(df_step, SLAS, 100, prb_win)
    assert scalars["rb_slot_conservation_abs_err"] == pytest.approx(0.0, abs=1e-9)
    assert scalars["rb_win_conservation_abs_err"] == pytest.approx(0.0, abs=1e-9)
    assert scalars["rb_win_scheduled_le_budget_violations"] == 0.0
    assert scalars["rb_win_multibeam_budget_gt_cap_windows"] == 0.0

    # Verify raw KPM history & summary CSVs for zero -999 and zero ghost/transient UEs
    import glob

    kpm_hist_matches = glob.glob(
        os.path.join(str(out_dir), "**", "sqr_kpm_history.csv"), recursive=True
    )
    summary_matches = glob.glob(
        os.path.join(str(out_dir), "**", "sqr_summary.csv"), recursive=True
    )
    assert len(kpm_hist_matches) == 1
    assert len(summary_matches) == 1

    kpm_df = pd.read_csv(kpm_hist_matches[0])
    sum_df = pd.read_csv(summary_matches[0])
    assert len(kpm_df) > 0
    assert not (kpm_df["snrDb"] == -999.0).any(), "Found legacy -999 sentinel in snrDb"

    # Every UE emitted in sqr_kpm_history must be attached==1 and have valid session >= 0.6 s
    sum_indexed = sum_df.set_index("ueId")
    for ue_id in kpm_df["ueImsi"].unique():
        row = sum_indexed.loc[ue_id]
        assert int(row["attached"]) == 1, f"Unattached UE {ue_id} appeared in KPM history"
        dep = float(row["departureSec"])
        arr = float(row["arrivalSec"])
        if dep >= 0.0:
            assert (dep - arr) >= 0.6 - 1e-6, f"Transient <0.6s UE {ue_id} appeared in KPM history"


def test_l3_anisotropic_kernel_and_het_controller_builder():
    """L3 per-slice action length scales, --het sla_threshold=0, and CerConfig dl_slots_per_win=80."""
    from nsoran.sqr_v2.cer_reallocator import (
        CerBlocoATelemetry,
        CerConfig,
        compute_mmtc_physical_floor,
    )
    from nsoran.sqr_v2.controller_v2 import build_controllers_v2, parse_action_length_scales_v2

    scales = parse_action_length_scales_v2("0.25,0.20,0.35", 3)
    assert scales == {0: pytest.approx(0.25), 1: pytest.approx(0.20), 2: pytest.approx(0.35)}
    assert parse_action_length_scales_v2(None, 3) == {0: 1.0, 1: 1.0, 2: 1.0}

    args_het_sqr = sqr_v2_run.parse_args(
        ["--controller", "sqr", "--het", "--action-length-scales", "0.25,0.20,0.35"]
    )
    ctls_sqr = build_controllers_v2(args_het_sqr)
    for sid, expected_ell in enumerate((0.25, 0.20, 0.35)):
        assert ctls_sqr[sid].sla_threshold == 0.0
        np.testing.assert_allclose(
            ctls_sqr[sid].kernel.length_scale,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, expected_ell]),
        )
        np.testing.assert_allclose(
            ctls_sqr[sid].regressor.kernel.length_scale,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, expected_ell]),
        )

    args_het_kbrl = sqr_v2_run.parse_args(["--controller", "kbrl", "--het"])
    ctls_kbrl = build_controllers_v2(args_het_kbrl)
    for sid in range(3):
        assert ctls_kbrl[sid].sla_threshold == 0.0

    # dl_slots_per_win propagation in Stage 1 mMTC physical floor
    cfg80 = CerConfig(dl_slots_per_win=80)
    assert cfg80.dl_slots_per_win == 80
    tel = CerBlocoATelemetry(
        n_ues=(1, 1, 2),
        queue_bytes=(0.0, 0.0, 40000.0),
        offered_mbps=(1.0, 0.5, 0.1),
        snr_db=(15.0, 15.0, 5.0),
    )
    floor50 = compute_mmtc_physical_floor(tel, 53, (True, True, True), dl_slots_per_win=50)
    floor80 = compute_mmtc_physical_floor(tel, 53, (True, True, True), dl_slots_per_win=80)
    assert floor80 <= floor50


def test_fase3_het_sqr_cer_smoke_with_tdd5ms_and_l3(tmp_path):
    """End-to-end heterogeneous SQR-CER + L3 + urllc_delayed_update smoke under TDD 5 ms."""
    import json

    out_dir = tmp_path / "fase3_het_cer_smoke"
    rc = sqr_v2_run.main(
        [
            "--controller", "sqr",
            "--gamma", "1",
            "--reallocator", "sqr_cer",
            "--het",
            "--empty-slice-guard",
            "--urllc-delayed-update",
            "--action-length-scales", "0.25,0.20,0.35",
            "--cer-dl-slots", "80",
            "--seed", "105",
            "--simTime", "6.0",
            "--cellRadius", "300",
            "--sliceLambdas", "12,24,12",
            "--sliceApps", "greedy,urllc,mmtc",
            "--warmup_s", "2.0",
            "--ipc-mode", "strict_rt",
            "--ipc-timeout-ms", "50",
            "--checkpoint_interval", "0",
            "--output", str(out_dir),
            "--run-id", "fase3_het_smoke_s105",
        ]
    )
    assert rc == 0

    het_csv = out_dir / "het_window_metrics.csv"
    assert het_csv.exists()
    df_het = pd.read_csv(het_csv)
    assert len(df_het) >= 40
    assert "sla_ok_all3" in df_het.columns

    done = json.loads((out_dir / "run_done.json").read_text(encoding="utf-8"))
    assert done["het"] is True
    assert done["ns3_exitcode"] == 0
    assert done["het_summary"]["n_post_warmup"] >= 20
    assert done["episode_summary"]["agent_decision_ms_p95"] < 20.0

    prb_win = load_prb_windows_df(str(out_dir))
    df_step = pd.read_csv(out_dir / "step_metrics.csv")
    scalars = compute_seed_scalars_v2(df_step, SLAS, 100, prb_win)
    assert scalars["rb_win_conservation_abs_err"] == pytest.approx(0.0, abs=1e-9)


def test_urllc_blackout_not_vacuous_when_tx_pkts_positive(tmp_path):
    """Blackout (urllc_tx_pkts > 0 and 0 delivered delay rows) must score SLA violation, not vacuity."""
    records = [
        {
            "window_index": 10, "window_seq": 10, "step": 10, "warmup": 0,
            "ipc_status": "APPLIED", "total_prbs": 53, "cer_stage": "HOASHI",
            "abstain_reason": "", "a_nom_0": 10, "a_nom_1": 10, "a_nom_2": 7,
            "a_exec_0": 15, "a_exec_1": 0, "a_exec_2": 7,
            "prb_sched_0": 15, "prb_sched_1": 0, "prb_sched_2": 7,
            "u_preq_0": "", "u_preq_1": "", "u_preq_2": "",
            "embb_thr_mbps": 5.0, "urllc_tx_pkts": 100,
            "urllc_p95_ms_online": None, "urllc_miss_pct_online": None,
            "mmtc_pdr_rolling": 1.0, "agent_wall_ms": 0.2,
        },
        {
            "window_index": 11, "window_seq": 11, "step": 11, "warmup": 0,
            "ipc_status": "APPLIED", "total_prbs": 53, "cer_stage": "HOASHI",
            "abstain_reason": "", "a_nom_0": 10, "a_nom_1": 0, "a_nom_2": 7,
            "a_exec_0": 15, "a_exec_1": 0, "a_exec_2": 7,
            "prb_sched_0": 15, "prb_sched_1": 0, "prb_sched_2": 7,
            "u_preq_0": "", "u_preq_1": "", "u_preq_2": "",
            "embb_thr_mbps": 5.0, "urllc_tx_pkts": 0,
            "urllc_p95_ms_online": None, "urllc_miss_pct_online": None,
            "mmtc_pdr_rolling": 1.0, "agent_wall_ms": 0.2,
        },
    ]
    out_csv = tmp_path / "het_blackout_test.csv"
    summary = sqr_v2_run._summarize_and_write_het_windows(None, records, output_csv_path=str(out_csv))
    df = pd.read_csv(out_csv).set_index("window_index")
    assert df.loc[10, "sla_ok_u"] == 0
    assert df.loc[10, "sla_ok_all3"] == 0
    assert df.loc[10, "margin_u"] == pytest.approx(-1.0)
    assert df.loc[10, "urllc_miss_pct"] == pytest.approx(100.0)
    assert df.loc[11, "sla_ok_u"] == 1
    assert df.loc[11, "sla_ok_all3"] == 1
    assert summary["sla_viol_urllc_pct"] == pytest.approx(50.0)
