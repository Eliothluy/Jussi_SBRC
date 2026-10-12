from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "run_controlled_rslaq_validation.sh"
DDQN = ROOT / "examples" / "rslaq_train_ddqn.py"
SAC = ROOT / "examples" / "rslaq_train_sac.py"
PREDICTIVE_SAC = ROOT / "examples" / "rslaq_train_predictive_sac.py"
PREDICTIVE_RUNNER = ROOT / "examples" / "run_predictive_rslaq_validation.sh"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _ddqn_text() -> str:
    return DDQN.read_text(encoding="utf-8")


def _sac_text() -> str:
    return SAC.read_text(encoding="utf-8")


def _predictive_sac_text() -> str:
    return PREDICTIVE_SAC.read_text(encoding="utf-8")


def _predictive_runner_text() -> str:
    return PREDICTIVE_RUNNER.read_text(encoding="utf-8")


class ControlledRunnerContractTest(unittest.TestCase):
    def test_controlled_runner_defaults_to_slice_aware_baselines(self):
        text = _runner_text()
        default_line = next(
            line for line in text.splitlines() if line.startswith("BASELINE_MODES=")
        )

        for mode in [
            "slice_weighted_pf",
            "psta_equal",
            "slice_weighted_rr",
            "slice_weighted_bcqi",
        ]:
            self.assertIn(mode, default_line)

    def test_controlled_runner_exports_rslaq_equivalent_baseline_metrics(self):
        text = _runner_text()

        self.assertIn("collect_baseline_runs", text)
        self.assertIn("baseline_class", text)
        self.assertIn("rslaq_violation_rate", text)
        self.assertIn("native_sla_satisfaction_pct", text)

    def test_predictive_sac_matches_non_terminal_sla_default(self):
        text = _predictive_sac_text()

        self.assertIn('"terminate_on_sla_violation": args.terminate_on_sla_violation', text)
        self.assertIn("parser.set_defaults(terminate_on_sla_violation=False)", text)
        self.assertIn('"--terminate-on-sla-violation"', text)

    def test_controlled_runner_exposes_predictive_sla_terminal_toggle(self):
        text = _runner_text()

        self.assertIn("PREDICTIVE_TERMINATE_ON_SLA_VIOLATION=", text)
        self.assertIn("predictive_terminal_args", text)
        self.assertIn("--terminate-on-sla-violation", text)
        self.assertIn('"predictive_terminate_on_sla_violation"', text)

    def test_predictive_runner_exposes_predictive_sla_terminal_toggle(self):
        text = _predictive_runner_text()

        self.assertIn("PREDICTIVE_TERMINATE_ON_SLA_VIOLATION=", text)
        self.assertIn("predictive_terminal_args", text)
        self.assertIn("--terminate-on-sla-violation", text)
        self.assertIn('"predictive_terminate_on_sla_violation"', text)

    def test_predictive_runner_uses_controlled_predictive_reward_aliases(self):
        text = _predictive_runner_text()

        self.assertIn("PREDICTIVE_REWARD_MODE=", text)
        self.assertIn("PREDICTIVE_P_STA_FRACTION=", text)
        self.assertIn('--reward_mode "${PREDICTIVE_REWARD_MODE}"', text)
        self.assertIn('--p_sta_static_fraction "${PREDICTIVE_P_STA_FRACTION}"', text)
        self.assertIn('"reward_mode": "${PREDICTIVE_REWARD_MODE}"', text)

    def test_predictive_runner_can_use_paper_faithful_outage(self):
        text = _predictive_runner_text()

        self.assertIn("PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE", text)
        self.assertIn("predictive_outage_args", text)
        self.assertIn("--paper-faithful-outage", text)
        self.assertIn('"demand_aware_embb_outage": ${PREDICTIVE_DEMAND_AWARE_EMBB_OUTAGE_JSON}', text)

    def test_training_scripts_expose_deterministic_evaluation_phase(self):
        for text in [_ddqn_text(), _sac_text(), _predictive_sac_text()]:
            self.assertIn("--eval_episodes", text)
            self.assertIn("--eval_seed_offset", text)
            self.assertIn("evaluation_metrics", text)
            self.assertIn("evaluation_log.csv", text)

        self.assertIn("epsilon=0", _ddqn_text())
        self.assertIn("deterministic=True", _sac_text())
        self.assertIn("deterministic=True", _predictive_sac_text())

    def test_controlled_runner_exposes_independent_training_replicates_and_eval(self):
        text = _runner_text()

        self.assertIn("TRAINING_REPLICATES=", text)
        self.assertIn("EVAL_EPISODES=", text)
        self.assertIn("EVAL_SEED_OFFSET=", text)
        self.assertIn("training_seed_for", text)
        self.assertIn("replicate", text)
        self.assertIn("--eval_episodes", text)
        self.assertIn("--eval_seed_offset", text)


if __name__ == "__main__":
    unittest.main()
