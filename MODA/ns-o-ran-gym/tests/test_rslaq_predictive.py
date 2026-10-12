"""Tests for RSLAQ predictive helpers."""

import csv
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.rslaq_predictive import (
    FEATURE_DIM,
    FORECAST_DIM,
    TemporalKpiForecaster,
    TemporalKpiForecasterLSTM,
    create_forecaster,
    build_forecast_sequences,
    forecast_from_history,
    frame_to_feature,
    load_baseline_frames,
    load_step_frames,
)


def _write_step_metrics(root, steps=8):
    sim_dir = os.path.join(root, "run")
    os.makedirs(sim_dir, exist_ok=True)
    path = os.path.join(sim_dir, "step_metrics.csv")
    fields = [
        "seed", "scenario", "episode", "step", "algo_mode", "slice_id",
        "slice", "throughput_mbps", "dTxBytes", "dRxBytes",
        "bufferBytes_mean", "bufferBytes_max", "plr_pct", "pdr_pct",
        "dLostPackets", "resourceSharePct", "action_embb",
        "action_urllc", "action_mtc", "scheduler_id", "scheduler_name",
        "raw_action", "reward", "outage_flag", "soft_flag",
        "terminated", "truncated", "sim_id",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(1, steps + 1):
            for sid, name in enumerate(["eMBB", "URLLC", "MTC"]):
                writer.writerow(
                    {
                        "seed": 1,
                        "scenario": "normal",
                        "episode": 1,
                        "step": step,
                        "algo_mode": "continuous",
                        "slice_id": sid,
                        "slice": name,
                        "throughput_mbps": 5.0 + sid,
                        "dTxBytes": 1000.0 * (sid + 1),
                        "dRxBytes": 900.0 * (sid + 1),
                        "bufferBytes_mean": 100.0 * (sid + 1),
                        "bufferBytes_max": 200.0 * (sid + 1),
                        "plr_pct": 0.0,
                        "pdr_pct": 100.0,
                        "dLostPackets": float(sid),
                        "resourceSharePct": 33.3,
                        "action_embb": 33.3,
                        "action_urllc": 40.0,
                        "action_mtc": 26.7,
                        "scheduler_id": -1,
                        "scheduler_name": "PF(default)",
                        "raw_action": "[0, 0, 0]",
                        "reward": 0.5,
                        "outage_flag": sid == 0 and step == steps,
                        "soft_flag": sid == 0 and step == steps - 1,
                        "terminated": step == steps,
                        "truncated": False,
                        "sim_id": "sim-a",
                    }
                )
    return path


def _write_baseline_result(root, steps=8):
    run_dir = os.path.join(root, "scenario=normal", "mode=slice_rr", "seed=1_run=1")
    os.makedirs(run_dir, exist_ok=True)
    timeseries_path = os.path.join(run_dir, "timeseries.csv")
    alloc_path = os.path.join(run_dir, "slice_alloc.csv")

    with open(timeseries_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_ms", "ue_id", "slice", "thr_mbps", "tx_bytes_delta",
                "rx_bytes_delta", "tx_packets_delta", "rx_packets_delta",
                "loss_pct_interval", "buffer_bytes", "dropped_packets_delta",
                "rsh_configured_pct",
            ],
        )
        writer.writeheader()
        for step in range(1, steps + 1):
            timestamp = step * 100
            for sid, name in enumerate(["eMBB", "URLLC", "MTC"]):
                writer.writerow(
                    {
                        "timestamp_ms": timestamp,
                        "ue_id": sid + 1,
                        "slice": name,
                        "thr_mbps": [12.0, 1.2, 2.0][sid],
                        "tx_bytes_delta": [4000, 500, 800][sid],
                        "rx_bytes_delta": [3800, 500, 760][sid],
                        "tx_packets_delta": 10,
                        "rx_packets_delta": 9,
                        "loss_pct_interval": 0.0,
                        "buffer_bytes": [100, 50, 20][sid],
                        "dropped_packets_delta": 0,
                        "rsh_configured_pct": 0.0,
                    }
                )

    with open(alloc_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp_ms", "slice", "rsh_real_pct"],
        )
        writer.writeheader()
        for step in range(1, steps + 1):
            timestamp = step * 100
            for sid, rsh in enumerate([50.0, 30.0, 20.0]):
                writer.writerow({"timestamp_ms": timestamp, "slice": sid, "rsh_real_pct": rsh})

    return timeseries_path


def test_load_step_frames_and_feature_shape():
    with tempfile.TemporaryDirectory() as root:
        path = _write_step_metrics(root)
        frames = load_step_frames(path)
    assert len(frames) == 8
    feature = frame_to_feature(frames[0])
    assert feature.shape == (FEATURE_DIM,)
    assert np.all(feature >= 0.0)
    assert np.all(feature <= 1.0)


def test_build_forecast_sequences_shape():
    with tempfile.TemporaryDirectory() as root:
        _write_step_metrics(root, steps=10)
        x, y, meta = build_forecast_sequences(root, sequence_len=4, horizon=2)
    assert x.shape[1:] == (4, FEATURE_DIM)
    assert y.shape[1:] == (FORECAST_DIM,)
    assert len(meta) == x.shape[0]


def test_load_baseline_frames_uses_slice_alloc_and_resource_reward():
    with tempfile.TemporaryDirectory() as root:
        path = _write_baseline_result(root)
        frames = load_baseline_frames(path)
    assert len(frames) == 8
    assert frames[0].scenario == "normal"
    assert frames[0].seed == 1
    assert np.allclose(frames[0].action, [0.5, 0.3, 0.2])
    assert frames[0].reward > 0.0


def test_build_forecast_sequences_auto_reads_baseline_results():
    with tempfile.TemporaryDirectory() as root:
        _write_baseline_result(root, steps=10)
        x, y, meta = build_forecast_sequences(root, sequence_len=4, horizon=2)
    assert x.shape[1:] == (4, FEATURE_DIM)
    assert y.shape[1:] == (FORECAST_DIM,)
    assert meta[0]["source_type"] == "baseline"


def test_forecaster_output_range():
    model = TemporalKpiForecaster(hidden_dim=8)
    x = torch.rand(2, 4, FEATURE_DIM)
    y = model(x)
    assert y.shape == (2, FORECAST_DIM)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_forecast_from_short_history_left_pads():
    model = TemporalKpiForecaster(hidden_dim=8)
    forecast = forecast_from_history(model, [], sequence_len=4, scenario="normal")
    assert forecast.shape == (FORECAST_DIM,)
    assert np.all(forecast >= 0.0)
    assert np.all(forecast <= 1.0)


def test_lstm_forecaster_output_range():
    model = TemporalKpiForecasterLSTM(hidden_dim=8, num_layers=1)
    x = torch.rand(2, 4, FEATURE_DIM)
    y = model(x)
    assert y.shape == (2, FORECAST_DIM)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_factory_creates_both_types():
    gru = create_forecaster("gru", hidden_dim=8)
    lstm = create_forecaster("lstm", hidden_dim=8, num_layers=1)
    assert isinstance(gru, TemporalKpiForecaster)
    assert isinstance(lstm, TemporalKpiForecasterLSTM)
    x = torch.rand(2, 4, FEATURE_DIM)
    assert gru(x).shape == (2, FORECAST_DIM)
    assert lstm(x).shape == (2, FORECAST_DIM)
