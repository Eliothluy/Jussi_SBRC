import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.offline_dataset import (
    SCENARIOS,
    build_dataset,
    extract_state_features,
    iter_heuristic_rows,
)
from nsoran.scoring import read_summary, score_summary_rows

SUMMARY_FIELDS = [
    "scenario", "baseline_mode", "slice", "throughput_mbps_mean",
    "delay_ms_mean", "pdr_pct", "sla_satisfaction_pct",
    "offered_load_satisfaction_pct", "budget_utilization_pct_mean",
    "buffer_bytes_mean",
]

METADATA = {
    "scenario": "normal",
    "num_ues": 20,
    "num_ues_per_slice": {"eMBB": 5, "URLLC": 5, "MTC": 10},
    "offered_load_mbps_per_slice": {"eMBB": 70.0, "URLLC": 1.0, "MTC": 2.0},
    "packet_size_bytes_per_slice": {"eMBB": 1500, "URLLC": 50, "MTC": 100},
}


def _write_summary(path: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for sl, thr in [("eMBB", "40"), ("URLLC", "1.0"), ("MTC", "2.0")]:
            writer.writerow({
                "scenario": "normal", "baseline_mode": mode, "slice": sl,
                "throughput_mbps_mean": thr, "delay_ms_mean": "12",
                "pdr_pct": "90", "sla_satisfaction_pct": "85",
                "offered_load_satisfaction_pct": "80",
                "budget_utilization_pct_mean": "75",
                "buffer_bytes_mean": "1000",
            })


def _write_metadata(path: Path, policy: str, weights: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(METADATA)
    meta["slice_weight_policy"] = policy
    meta["slice_weights_configured"] = weights
    path.write_text(json.dumps(meta), encoding="utf-8")


def _write_candidate(path: Path, score: float, weights: list, eval_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "method": "ga", "evaluation_id": eval_id, "scenario": "normal",
        "seed": 1, "run": 1, "weights": weights, "score": score,
    }), encoding="utf-8")


def _write_slice_alloc(path: Path, weights_by_step: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ms", "slice", "configured_weight", "budget_rbg"])
        for step, weights in enumerate(weights_by_step):
            for slice_id, weight in enumerate(weights):
                writer.writerow([400 + 10 * step, slice_id, f"{weight:.4f}", 100])


class OfflineDatasetFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.meta_root = root / "metaheuristics"
        self.heur_root = root / "heuristics_ns3" / "results_rslaq_network_only"

        # Scenario metadata reachable from the metaheuristics tree.
        (self.meta_root / "scenario=normal").mkdir(parents=True)
        _write_metadata(self.meta_root / "scenario=normal" / "metadata.json",
                        "static", [0.3333, 0.4, 0.2667])

        evals = self.meta_root / "scenario=normal/seed=1/metaheuristic_search/evals"
        _write_candidate(evals / "ga_eval_0001/candidate.json",
                         62.4, [0.2, 0.3, 0.5], 1)
        # Crash-tolerated eval: must be skipped.
        _write_candidate(evals / "ga_eval_0002/candidate.json",
                         -1e6, [0.1, 0.1, 0.8], 2)

        # Static heuristic mode.
        static_dir = self.heur_root / "scenario=normal/mode=slice_weighted_pf/seed=1_run=1"
        _write_metadata(static_dir / "metadata.json", "static", [0.3333, 0.4, 0.2667])
        _write_summary(static_dir / "summary.csv", "slice_weighted_pf")

        # Dynamic heuristic mode: weights vary per window.
        self.dynamic_steps = [[0.8, 0.1, 0.1], [0.4, 0.3, 0.3], [0.6, 0.2, 0.2]]
        dyn_dir = self.heur_root / "scenario=normal/mode=slice_aqps/seed=1_run=1"
        _write_metadata(dyn_dir / "metadata.json", "aqps", [0.3333, 0.3333, 0.3333])
        _write_summary(dyn_dir / "summary.csv", "slice_aqps")
        _write_slice_alloc(dyn_dir / "slice_alloc.csv", self.dynamic_steps)
        self.dyn_summary = dyn_dir / "summary.csv"

        # Pure mode: must always be excluded.
        pure_dir = self.heur_root / "scenario=normal/mode=pure_rr/seed=1_run=1"
        _write_metadata(pure_dir / "metadata.json", "NA", [])
        _write_summary(pure_dir / "summary.csv", "pure_rr")

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, **kwargs):
        return build_dataset(self.meta_root, self.heur_root,
                             scenarios=["normal"], seeds=[1], **kwargs)


class BuildDatasetTest(OfflineDatasetFixture):
    def test_row_counts_and_skips(self):
        df, stats = self.build()
        self.assertEqual(len(df), 3)  # 1 meta + static + dynamic
        self.assertEqual(stats.get("skipped_crash"), 1)
        self.assertEqual(stats.get("skipped_pure"), 1)
        self.assertEqual(sorted(df["source"]), ["heuristic", "heuristic", "metaheuristic"])

    def test_provenance_and_mode_columns(self):
        df, _ = self.build()
        by_mode = df.set_index("mode")
        self.assertEqual(by_mode.loc["ga", "weight_provenance"], "search")
        self.assertEqual(by_mode.loc["slice_weighted_pf", "weight_provenance"], "configured")
        self.assertEqual(by_mode.loc["slice_aqps", "weight_provenance"], "time_averaged")

    def test_dynamic_action_is_renormalized_time_average(self):
        df, _ = self.build()
        row = df[df["mode"] == "slice_aqps"].iloc[0]
        n = len(self.dynamic_steps)
        expected = [sum(step[i] for step in self.dynamic_steps) / n for i in range(3)]
        total = sum(expected)
        expected = [v / total for v in expected]
        for value, exp in zip([row["w_embb"], row["w_urllc"], row["w_mtc"]], expected):
            self.assertAlmostEqual(value, exp, places=4)

    def test_actions_on_simplex_and_one_hot(self):
        df, _ = self.build()
        sums = df[["w_embb", "w_urllc", "w_mtc"]].sum(axis=1)
        for total in sums:
            self.assertAlmostEqual(total, 1.0, places=3)
        for sc in SCENARIOS:
            self.assertIn(f"is_{sc}", df.columns)
        self.assertTrue((df["is_normal"] == 1).all())
        self.assertTrue((df["is_congestion"] == 0).all())

    def test_exclude_dynamic_modes(self):
        df, stats = self.build(include_dynamic_modes=False)
        self.assertEqual(len(df), 2)
        self.assertEqual(stats.get("skipped_dynamic"), 1)
        self.assertNotIn("slice_aqps", set(df["mode"]))

    def test_no_heuristics(self):
        df, _ = self.build(include_heuristics=False)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["source"], "metaheuristic")

    def test_heuristic_score_matches_scoring_module(self):
        """The builder must not introduce its own scoring."""
        rows = list(iter_heuristic_rows(self.heur_root, ["normal"], [1]))
        expected = score_summary_rows(read_summary(self.dyn_summary))
        by_mode = {r["mode"]: r for r in rows}
        self.assertAlmostEqual(by_mode["slice_aqps"]["score"], expected, places=9)

    def test_state_features_from_metadata(self):
        features = extract_state_features(METADATA)
        self.assertEqual(features["num_ues_total"], 20)
        self.assertEqual(features["num_ues_mtc"], 10)
        self.assertAlmostEqual(features["offered_load_embb"], 70.0)
        df, _ = self.build()
        self.assertTrue((df["num_ues_total"] == 20).all())

    def test_state_features_list_metadata(self):
        meta = dict(METADATA)
        meta["num_ues_per_slice"] = [5, 5, 10]
        meta["offered_load_mbps_per_slice"] = [70.0, 1.0, 2.0]
        meta["packet_size_bytes_per_slice"] = [1500, 50, 100]
        features = extract_state_features(meta)
        self.assertEqual(features["num_ues_urllc"], 5)
        self.assertEqual(features["pkt_size_urllc"], 50)


if __name__ == "__main__":
    unittest.main()
