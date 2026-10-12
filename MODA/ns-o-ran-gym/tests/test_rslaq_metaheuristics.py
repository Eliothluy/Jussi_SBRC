import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "run_rslaq_metaheuristics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("rslaq_metaheuristics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RslaqMetaheuristicsTest(unittest.TestCase):
    def test_normalize_weights_projects_to_non_negative_simplex(self):
        module = load_module()

        self.assertEqual(module.normalize_weights([2.0, -1.0, 2.0]), [0.5, 0.0, 0.5])
        self.assertEqual(module.normalize_weights([0.0, 0.0, 0.0]), [1.0 / 3.0] * 3)

    def test_score_penalizes_starved_mtc_even_with_high_total_throughput(self):
        module = load_module()

        balanced = [
            {"slice": "eMBB", "throughput_mbps_mean": "45", "sla_satisfaction_pct": "90", "offered_load_satisfaction_pct": "70", "pdr_pct": "80", "delay_ms_mean": "200", "buffer_bytes_mean": "2000000", "budget_utilization_pct_mean": "100"},
            {"slice": "URLLC", "throughput_mbps_mean": "4", "sla_satisfaction_pct": "100", "offered_load_satisfaction_pct": "90", "pdr_pct": "90", "delay_ms_mean": "5", "buffer_bytes_mean": "1000", "budget_utilization_pct_mean": "100"},
            {"slice": "MTC", "throughput_mbps_mean": "30", "sla_satisfaction_pct": "100", "offered_load_satisfaction_pct": "70", "pdr_pct": "85", "delay_ms_mean": "50", "buffer_bytes_mean": "100000", "budget_utilization_pct_mean": "100"},
        ]
        starved = [
            {"slice": "eMBB", "throughput_mbps_mean": "110", "sla_satisfaction_pct": "100", "offered_load_satisfaction_pct": "80", "pdr_pct": "85", "delay_ms_mean": "100", "buffer_bytes_mean": "1000000", "budget_utilization_pct_mean": "100"},
            {"slice": "URLLC", "throughput_mbps_mean": "4", "sla_satisfaction_pct": "100", "offered_load_satisfaction_pct": "90", "pdr_pct": "90", "delay_ms_mean": "5", "buffer_bytes_mean": "1000", "budget_utilization_pct_mean": "100"},
            {"slice": "MTC", "throughput_mbps_mean": "0", "sla_satisfaction_pct": "0", "offered_load_satisfaction_pct": "0", "pdr_pct": "0", "delay_ms_mean": "0", "buffer_bytes_mean": "0", "budget_utilization_pct_mean": "0"},
        ]

        self.assertGreater(module.score_summary_rows(balanced), module.score_summary_rows(starved))

    def test_build_sim_command_uses_slice_custom_weights(self):
        module = load_module()

        command = module.build_sim_command(
            scenario="normal",
            output_root=Path("/tmp/rslaq-meta"),
            weights=[0.7, 0.05, 0.25],
            intra_algo="PF",
            sim_time=3.0,
            app_start=0.4,
            drain_time=0.2,
            period_ms=10,
            seed=2,
            run=1,
        )

        self.assertIn("--baselineMode=slice_custom", command)
        self.assertIn("--weights=0.700000,0.050000,0.250000", command)
        self.assertIn("--scenario=normal", command)

    def test_seed_weight_candidates_returns_simplex_priors(self):
        module = load_module()

        candidates = module.seed_weight_candidates(module.random.Random(7), 5)

        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0], [1.0 / 3.0] * 3)
        for weights in candidates:
            self.assertAlmostEqual(sum(weights), 1.0)
            self.assertTrue(all(weight >= 0.0 for weight in weights))

    def test_optimize_hybrid_uses_hybrid_method_and_preserves_simplex(self):
        module = load_module()
        calls = []
        crossover_calls = []
        original_crossover = module.crossover_weights

        def counting_crossover(a, b, rng):
            crossover_calls.append((list(a), list(b)))
            return original_crossover(a, b, rng)

        module.crossover_weights = counting_crossover

        def evaluate(method, evaluation_id, weights):
            normalized = module.normalize_weights(weights)
            calls.append((method, evaluation_id, normalized))
            score = 100.0 - sum(abs(a - b) for a, b in zip(normalized, [0.65, 0.10, 0.25]))
            return module.Evaluation(
                method=method,
                evaluation_id=evaluation_id,
                scenario="normal",
                seed=1,
                run=1,
                weights=normalized,
                score=score,
                result_dir=Path("/tmp/result"),
                log_file=Path("/tmp/log"),
            )

        evaluations = module.optimize_hybrid(
            module.random.Random(11),
            evaluate,
            iterations=2,
            population_size=6,
            mutation_strength=0.10,
        )

        self.assertEqual(len(evaluations), 14)
        self.assertEqual([call[1] for call in calls], list(range(1, 15)))
        self.assertTrue(all(call[0] == "hybrid" for call in calls))
        self.assertGreaterEqual(len(crossover_calls), 1)
        for evaluation in evaluations:
            self.assertAlmostEqual(sum(evaluation.weights), 1.0)
            self.assertTrue(all(weight >= 0.0 for weight in evaluation.weights))

    def test_parse_args_accepts_multi_seed_and_multi_scenario(self):
        module = load_module()
        import sys as _sys

        original = _sys.argv
        _sys.argv = [
            "run_rslaq_metaheuristics.py",
            "--method", "all",
            "--scenarios", "normal congestion",
            "--seeds", "1 2 3",
            "--iterations", "2",
        ]
        try:
            args = module.parse_args()
        finally:
            _sys.argv = original

        self.assertEqual(args.scenarios, ["normal", "congestion"])
        self.assertEqual(args.seeds, [1, 2, 3])

    def test_best_candidate_filename_includes_scenario_and_seed(self):
        module = load_module()

        name = module.best_candidate_filename("congestion", 2)
        self.assertIn("congestion", name)
        self.assertIn("seed2", name)
        self.assertTrue(name.endswith(".json"))


if __name__ == "__main__":
    unittest.main()
