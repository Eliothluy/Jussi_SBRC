import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    import numpy as np

    from nsoran.offline_models import (
        ALGO_DESCRIPTIONS,
        STATE_COLS,
        ActorNet,
        PolicyNet,
        QNet,
        build_simplex_actions,
        load_model,
        nearest_action_index,
        normalize_state,
        predict_weights,
        save_checkpoint,
    )


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class SimplexActionsTest(unittest.TestCase):
    def test_table_size_and_validity(self):
        table = build_simplex_actions(step=0.1)
        self.assertEqual(len(table), 66)
        self.assertTrue((table >= 0).all())
        for row in table:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=5)

    def test_nearest_action_index(self):
        table = build_simplex_actions(step=0.1)
        idx = nearest_action_index(np.array([0.31, 0.39, 0.30]), table)
        np.testing.assert_allclose(table[idx], [0.3, 0.4, 0.3], atol=1e-6)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class CheckpointRoundTripTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.state_dim = len(STATE_COLS)
        self.norm_params = {
            "min": {c: 0.0 for c in STATE_COLS},
            "range": {c: 1.0 for c in STATE_COLS},
        }
        self.x = np.random.RandomState(0).rand(4, self.state_dim).astype(np.float32)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _round_trip(self, model, model_type, action_table=None):
        path = self.tmp / f"{model_type}.pt"
        save_checkpoint(
            path, model, model_type=model_type, state_dim=self.state_dim,
            norm_params=self.norm_params, loss_history=[1.0, 0.5],
            train_seed=42, dataset_version="v2", action_table=action_table,
        )
        loaded, ckpt = load_model(path, model_type)
        before = predict_weights(model, model_type, self.x, action_table)
        after = predict_weights(loaded, model_type, self.x,
                                ckpt.get("action_table", action_table))
        np.testing.assert_allclose(before, after, atol=1e-6)
        self.assertEqual(ckpt["algo_description"], ALGO_DESCRIPTIONS[model_type])
        self.assertEqual(ckpt["norm_params"], self.norm_params)
        self.assertEqual(ckpt["train_seed"], 42)
        self.assertEqual(ckpt["dataset_version"], "v2")
        return after

    def test_ddqn_round_trip(self):
        table = build_simplex_actions(0.1)
        preds = self._round_trip(QNet(self.state_dim, len(table)), "ddqn", table)
        for row in preds:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)

    def test_sac_round_trip(self):
        preds = self._round_trip(ActorNet(self.state_dim), "sac")
        for row in preds:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)
            self.assertTrue((row >= 0).all())

    def test_ppo_round_trip(self):
        preds = self._round_trip(PolicyNet(self.state_dim), "ppo")
        for row in preds:
            self.assertAlmostEqual(float(row.sum()), 1.0, places=4)
            self.assertTrue((row >= 0).all())

    def test_ddqn_requires_action_table(self):
        with self.assertRaises(ValueError):
            save_checkpoint(
                self.tmp / "bad.pt", QNet(self.state_dim, 66),
                model_type="ddqn", state_dim=self.state_dim,
                norm_params=self.norm_params, loss_history=[], train_seed=0,
            )


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class NormalizeStateTest(unittest.TestCase):
    def test_normalize_state(self):
        params = {
            "min": {c: 10.0 for c in STATE_COLS},
            "range": {c: 20.0 for c in STATE_COLS},
        }
        raw = np.full((1, len(STATE_COLS)), 20.0)
        norm = normalize_state(raw, params)
        np.testing.assert_allclose(norm, 0.5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
