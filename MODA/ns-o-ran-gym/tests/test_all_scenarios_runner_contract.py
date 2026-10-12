from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "run_all_scenarios.sh"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


class AllScenariosRunnerContractTest(unittest.TestCase):
    def test_runner_defaults_to_current_paper_baseline_families(self):
        text = _runner_text()

        self.assertIn("SCHEDULER_MODES=", text)
        for mode in ["pure_rr", "pure_bcqi", "pure_pf"]:
            self.assertIn(mode, text)

        self.assertIn("RSLAQ_MODES=", text)
        for mode in ["slice_weighted_pf", "psta_equal"]:
            self.assertIn(mode, text)

        self.assertIn("AQPS_MODE=", text)
        self.assertIn("slice_aqps", text)

        self.assertIn("HEURISTIC_MODES=", text)
        for mode in [
            "slice_demand_greedy",
            "slice_sla_greedy",
            "slice_least_waste",
            "slice_qos_mixed",
            "slice_random_vine",
            "slice_meta_risk_elastic",
        ]:
            self.assertIn(mode, text)

    def test_runner_defaults_to_all_current_metaheuristics(self):
        text = _runner_text()

        self.assertIn("META_METHOD=\"${META_METHOD:-all}\"", text)
        self.assertIn("--method \"${META_METHOD}\"", text)

    def test_runner_defines_group1_and_group2_for_fair_comparison(self):
        text = _runner_text()

        self.assertIn("GROUP1_MODES=", text)
        self.assertIn("GROUP2_MODES=", text)

    def test_runner_defines_rslaq_reference_with_fixed_weights(self):
        text = _runner_text()

        self.assertIn("RSLAQ_REFERENCE_WEIGHTS=", text)
        self.assertIn("0.3333,0.4000,0.2667", text)

    def test_runner_has_meta_evaluation_phase(self):
        text = _runner_text()

        self.assertIn("RUN_META_EVALUATION", text)
        self.assertIn("run_meta_evaluation.py", text)

    def test_runner_passes_multi_seed_to_metaheuristic_search(self):
        text = _runner_text()

        self.assertIn("--seeds", text)
        self.assertIn("--scenarios", text)


if __name__ == "__main__":
    unittest.main()
