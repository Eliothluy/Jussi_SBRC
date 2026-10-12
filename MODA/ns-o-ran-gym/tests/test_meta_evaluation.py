import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "run_meta_evaluation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("meta_evaluation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MetaEvaluationTest(unittest.TestCase):
    def test_load_best_candidates_finds_per_scenario_seed_files(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for scenario in ["normal", "congestion"]:
                for seed in [1, 2]:
                    search_dir = root / f"scenario={scenario}" / f"seed={seed}" / "metaheuristic_search"
                    search_dir.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "method": "hybrid",
                        "scenario": scenario,
                        "seed": seed,
                        "score": 42.0,
                        "weights": {"eMBB": 0.5, "URLLC": 0.3, "MTC": 0.2},
                        "ns3_args": {"baselineMode": "slice_custom", "weights": "0.500000,0.300000,0.200000"},
                    }
                    (search_dir / f"best_candidate_{scenario}_seed{seed}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )

            candidates = module.load_best_candidates(root)

            self.assertEqual(len(candidates), 4)
            keys = {(c["scenario"], c["seed"]) for c in candidates}
            self.assertIn(("normal", 1), keys)
            self.assertIn(("congestion", 2), keys)

    def test_build_eval_command_uses_slice_custom_with_best_weights(self):
        module = load_module()

        cmd = module.build_eval_command(
            scenario="congestion",
            output_root=Path("/tmp/eval"),
            weights_str="0.500000,0.300000,0.200000",
            intra_algo="PF",
            sim_time=20.0,
            app_start=0.4,
            drain_time=0.2,
            period_ms=10,
            seed=2,
            run=1,
        )

        self.assertIn("--baselineMode=slice_custom", cmd)
        self.assertIn("--weights=0.500000,0.300000,0.200000", cmd)
        self.assertIn("--scenario=congestion", cmd)
        self.assertIn("--seed=2", cmd)
        self.assertIn("--intraAlgo=PF", cmd)
        self.assertIn("--simTime=20", cmd)


if __name__ == "__main__":
    unittest.main()
