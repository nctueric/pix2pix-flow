"""Tests for rl_scheduler.py"""

import os
import json
import tempfile
import unittest
import numpy as np

from rl_scheduler import HyperparameterScheduler


class TestHyperparameterSchedulerInit(unittest.TestCase):
    """Test initialization and default state."""

    def test_initial_hyperparams_match_base(self):
        s = HyperparameterScheduler(0.001, 1.0, 2.0, seed=0)
        hp = s.get_current_hyperparams()
        self.assertEqual(hp['lr'], 0.001)
        self.assertEqual(hp['code_loss_scale'], 1.0)
        self.assertEqual(hp['mle_loss_scale'], 2.0)

    def test_initial_policy_weights_are_zero(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        np.testing.assert_array_equal(s.W_mean, np.zeros((3, 9)))
        np.testing.assert_array_equal(s.b_mean, np.zeros(3))

    def test_initial_value_weights_are_zero(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        np.testing.assert_array_equal(s.W_value, np.zeros(9))
        self.assertEqual(s.b_value, 0.0)

    def test_initial_log_std(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        np.testing.assert_array_equal(s.log_std, np.full(3, -1.0))

    def test_set_total_epochs(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        s.set_total_epochs(100)
        self.assertEqual(s._total_epochs, 100)

    def test_set_total_epochs_zero_clamped(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        s.set_total_epochs(0)
        self.assertEqual(s._total_epochs, 1)

    def test_epoch_count_starts_zero(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        self.assertEqual(s._epoch_count, 0)

    def test_prev_val_loss_starts_none(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        self.assertIsNone(s._prev_val_loss)


class TestStepBasics(unittest.TestCase):
    """Test that step() returns valid output and updates state."""

    def setUp(self):
        self.s = HyperparameterScheduler(
            0.001, 1.0, 1.0, seed=42, action_bound=0.3)

    def test_step_returns_dict_with_required_keys(self):
        result = self.s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        self.assertIn('lr', result)
        self.assertIn('code_loss_scale', result)
        self.assertIn('mle_loss_scale', result)

    def test_step_returns_float_values(self):
        result = self.s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        for key in ['lr', 'code_loss_scale', 'mle_loss_scale']:
            self.assertIsInstance(result[key], float)

    def test_step_increments_epoch_count(self):
        self.assertEqual(self.s._epoch_count, 0)
        self.s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        self.assertEqual(self.s._epoch_count, 1)
        self.s.step(4.5, 4.5, 0.9, 4.2, 4.2)
        self.assertEqual(self.s._epoch_count, 2)

    def test_step_updates_prev_val_loss(self):
        self.assertIsNone(self.s._prev_val_loss)
        self.s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        self.assertAlmostEqual(self.s._prev_val_loss, 9.0)

    def test_first_step_reward_is_zero(self):
        """First step has no previous val loss, so reward should be 0."""
        self.s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        # No reward stored yet (needs a previous action)
        self.assertEqual(len(self.s._rewards), 0)

    def test_second_step_stores_reward(self):
        self.s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        self.s.step(4.5, 4.5, 0.9, 4.0, 4.0)
        # Reward = prev(9.0) - current(8.0) = 1.0
        self.assertEqual(len(self.s._rewards), 1)
        self.assertAlmostEqual(self.s._rewards[0], 1.0)

    def test_hyperparams_change_after_step(self):
        initial = self.s.get_current_hyperparams()
        self.s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        updated = self.s.get_current_hyperparams()
        # With non-zero seed, at least one value should differ
        changed = any(initial[k] != updated[k] for k in initial)
        self.assertTrue(changed)

    def test_get_current_hyperparams_matches_step_return(self):
        result = self.s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        current = self.s.get_current_hyperparams()
        self.assertEqual(result, current)


class TestClamping(unittest.TestCase):
    """Test that hyperparameters respect their range constraints."""

    def test_lr_clamped_to_range(self):
        s = HyperparameterScheduler(
            0.001, 1.0, 1.0, seed=0,
            lr_range=(1e-4, 5e-3),
            action_bound=5.0)  # large bound to force clamping
        for _ in range(50):
            s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        hp = s.get_current_hyperparams()
        self.assertGreaterEqual(hp['lr'], 1e-4)
        self.assertLessEqual(hp['lr'], 5e-3)

    def test_code_scale_clamped_to_range(self):
        s = HyperparameterScheduler(
            0.001, 1.0, 1.0, seed=0,
            code_scale_range=(0.5, 2.0),
            action_bound=5.0)
        for _ in range(50):
            s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        hp = s.get_current_hyperparams()
        self.assertGreaterEqual(hp['code_loss_scale'], 0.5)
        self.assertLessEqual(hp['code_loss_scale'], 2.0)

    def test_mle_scale_clamped_to_range(self):
        s = HyperparameterScheduler(
            0.001, 1.0, 1.0, seed=0,
            mle_scale_range=(0.1, 10.0),
            action_bound=5.0)
        for _ in range(50):
            s.step(5.0, 5.0, 1.0, 4.5, 4.5)
        hp = s.get_current_hyperparams()
        self.assertGreaterEqual(hp['mle_loss_scale'], 0.1)
        self.assertLessEqual(hp['mle_loss_scale'], 10.0)

    def test_all_hyperparams_stay_positive(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=7, action_bound=0.5)
        for _ in range(100):
            hp = s.step(
                np.random.uniform(1, 10),
                np.random.uniform(1, 10),
                np.random.uniform(0, 2),
                np.random.uniform(1, 10),
                np.random.uniform(1, 10))
            self.assertGreater(hp['lr'], 0)
            self.assertGreater(hp['code_loss_scale'], 0)
            self.assertGreater(hp['mle_loss_scale'], 0)


class TestReproducibility(unittest.TestCase):
    """Test that the same seed produces identical trajectories."""

    def test_same_seed_same_trajectory(self):
        results_a = []
        results_b = []
        for results, seed in [(results_a, 42), (results_b, 42)]:
            s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=seed)
            s.set_total_epochs(100)
            for i in range(20):
                r = s.step(5.0 - i * 0.1, 5.0 - i * 0.1,
                           1.0 - i * 0.02, 4.5 - i * 0.1, 4.5 - i * 0.1)
                results.append(r)
        for a, b in zip(results_a, results_b):
            self.assertAlmostEqual(a['lr'], b['lr'], places=12)
            self.assertAlmostEqual(a['code_loss_scale'], b['code_loss_scale'], places=12)
            self.assertAlmostEqual(a['mle_loss_scale'], b['mle_loss_scale'], places=12)

    def test_different_seed_different_trajectory(self):
        def run(seed):
            s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=seed)
            results = []
            for _ in range(5):
                results.append(s.step(5.0, 5.0, 1.0, 4.5, 4.5))
            return results

        a = run(0)
        b = run(99)
        differs = any(
            a[i]['lr'] != b[i]['lr'] for i in range(len(a)))
        self.assertTrue(differs)


class TestRewardComputation(unittest.TestCase):
    """Test the reward signal logic."""

    def test_improving_val_loss_gives_positive_reward(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        s.step(5.0, 5.0, 1.0, 4.0, 4.0)  # val_loss = 8.0
        s.step(4.0, 4.0, 0.8, 3.0, 3.0)  # val_loss = 6.0, improvement = 2.0
        self.assertGreater(s._rewards[0], 0)

    def test_worsening_val_loss_gives_negative_reward(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        s.step(5.0, 5.0, 1.0, 4.0, 4.0)  # val_loss = 8.0
        s.step(6.0, 6.0, 1.2, 5.0, 5.0)  # val_loss = 10.0, worsened
        self.assertLess(s._rewards[0], 0)

    def test_flat_val_loss_gives_zero_reward(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        s.step(5.0, 5.0, 1.0, 4.0, 4.0)  # val_loss = 8.0
        s.step(5.0, 5.0, 1.0, 4.0, 4.0)  # val_loss = 8.0
        self.assertAlmostEqual(s._rewards[0], 0.0)


class TestPolicyUpdate(unittest.TestCase):
    """Test that _update_policy runs correctly after enough steps."""

    def test_update_triggers_after_10_rewards(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        W_before = s.W_mean.copy()
        # Need 11 steps to get 10 rewards (first step has no reward)
        for i in range(12):
            s.step(5.0 - i * 0.1, 5.0 - i * 0.1,
                   1.0, 4.5 - i * 0.1, 4.5 - i * 0.1)
        # After update, policy weights should change
        # (unless all advantages are exactly zero, which is unlikely)
        self.assertFalse(np.allclose(s.W_mean, W_before))

    def test_rewards_cleared_after_update(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        # 11 steps = 10 rewards (first step has no reward).
        # Update triggers when len(rewards) >= 10, which is at step 11.
        for i in range(11):
            s.step(5.0 - i * 0.1, 5.0 - i * 0.1,
                   1.0, 4.5 - i * 0.1, 4.5 - i * 0.1)
        # Rewards buffer should be cleared after update
        self.assertEqual(len(s._rewards), 0)

    def test_log_std_stays_in_bounds(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0, policy_lr=1.0)
        for i in range(50):
            s.step(5.0 - i * 0.05, 5.0 - i * 0.05,
                   1.0, 4.5 - i * 0.05, 4.5 - i * 0.05)
        self.assertTrue(np.all(s.log_std >= -3.0))
        self.assertTrue(np.all(s.log_std <= 1.0))


class TestDiscountedReturns(unittest.TestCase):
    """Test _compute_returns directly."""

    def test_single_reward(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, discount=0.9)
        returns = s._compute_returns(np.array([5.0]))
        np.testing.assert_array_almost_equal(returns, [5.0])

    def test_two_rewards(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, discount=0.9)
        returns = s._compute_returns(np.array([1.0, 2.0]))
        # returns[1] = 2.0
        # returns[0] = 1.0 + 0.9 * 2.0 = 2.8
        np.testing.assert_array_almost_equal(returns, [2.8, 2.0])

    def test_three_rewards(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, discount=0.5)
        returns = s._compute_returns(np.array([1.0, 2.0, 4.0]))
        # returns[2] = 4.0
        # returns[1] = 2.0 + 0.5 * 4.0 = 4.0
        # returns[0] = 1.0 + 0.5 * 4.0 = 3.0
        np.testing.assert_array_almost_equal(returns, [3.0, 4.0, 4.0])

    def test_zero_discount(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, discount=0.0)
        rewards = np.array([1.0, 2.0, 3.0])
        returns = s._compute_returns(rewards)
        np.testing.assert_array_almost_equal(returns, rewards)


class TestStateNormalization(unittest.TestCase):
    """Test the online Welford normalization."""

    def test_first_state_normalized_to_zero(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
        normed = s._normalize_state(state)
        # After 1 sample, mean = state, so (state - mean) = 0
        np.testing.assert_array_almost_equal(normed, np.zeros(9))

    def test_normalization_count_increments(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        self.assertEqual(s._state_count, 0)
        s._normalize_state(np.ones(9))
        self.assertEqual(s._state_count, 1)
        s._normalize_state(np.ones(9))
        self.assertEqual(s._state_count, 2)

    def test_normalization_with_constant_input(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        for _ in range(10):
            normed = s._normalize_state(np.ones(9) * 5.0)
        # Mean should converge to 5.0, variance near 0 -> clamped to 1e-8
        np.testing.assert_array_almost_equal(
            s._state_mean, np.ones(9) * 5.0, decimal=5)


class TestBuildState(unittest.TestCase):
    """Test state vector construction."""

    def test_state_shape(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        s.set_total_epochs(100)
        state = s._build_state(5.0, 4.0, 1.0, 3.0, 2.0)
        self.assertEqual(state.shape, (9,))

    def test_state_values(self):
        s = HyperparameterScheduler(0.001, 1.0, 2.0, seed=0)
        s.set_total_epochs(100)
        s._epoch_count = 50
        state = s._build_state(5.0, 4.0, 1.0, 3.0, 2.0)
        expected = np.array([
            5.0,   # train_loss_A
            4.0,   # train_loss_B
            1.0,   # code_loss
            3.0,   # val_loss_A
            2.0,   # val_loss_B
            1.0,   # lr_ratio = 0.001 / 0.001
            1.0,   # current_code_loss_scale
            2.0,   # current_mle_loss_scale
            0.5,   # epoch_progress = 50 / 100
        ])
        np.testing.assert_array_almost_equal(state, expected)


class TestSaveLoad(unittest.TestCase):
    """Test save/load round-trip preserves all state."""

    def test_save_creates_file(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            s.save(path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIn('W_mean', data)
            self.assertIn('current_lr', data)
        finally:
            os.unlink(path)

    def test_load_restores_hyperparams(self):
        s1 = HyperparameterScheduler(0.001, 1.0, 1.0, seed=42)
        for i in range(5):
            s1.step(5.0 - i * 0.2, 5.0 - i * 0.2,
                    1.0, 4.5 - i * 0.2, 4.5 - i * 0.2)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            s1.save(path)
            s2 = HyperparameterScheduler(0.001, 1.0, 1.0)
            s2.load(path)
            self.assertEqual(
                s1.get_current_hyperparams(),
                s2.get_current_hyperparams())
        finally:
            os.unlink(path)

    def test_load_restores_policy_weights(self):
        s1 = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        # Run enough steps to trigger a policy update
        for i in range(15):
            s1.step(5.0 - i * 0.1, 5.0 - i * 0.1,
                    1.0, 4.5 - i * 0.1, 4.5 - i * 0.1)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            s1.save(path)
            s2 = HyperparameterScheduler(0.001, 1.0, 1.0)
            s2.load(path)
            np.testing.assert_array_equal(s1.W_mean, s2.W_mean)
            np.testing.assert_array_equal(s1.b_mean, s2.b_mean)
            np.testing.assert_array_equal(s1.log_std, s2.log_std)
            np.testing.assert_array_equal(s1.W_value, s2.W_value)
            self.assertEqual(s1.b_value, s2.b_value)
        finally:
            os.unlink(path)

    def test_load_restores_normalization_state(self):
        s1 = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        for i in range(5):
            s1.step(5.0, 5.0, 1.0, 4.5, 4.5)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            s1.save(path)
            s2 = HyperparameterScheduler(0.001, 1.0, 1.0)
            s2.load(path)
            np.testing.assert_array_equal(s1._state_mean, s2._state_mean)
            np.testing.assert_array_equal(s1._state_var, s2._state_var)
            self.assertEqual(s1._state_count, s2._state_count)
            self.assertEqual(s1._prev_val_loss, s2._prev_val_loss)
            self.assertEqual(s1._epoch_count, s2._epoch_count)
        finally:
            os.unlink(path)


class TestActionMechanics(unittest.TestCase):
    """Test low-level action sampling and application."""

    def test_action_clipped_to_bound(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0, action_bound=0.1)
        # Force large policy output
        s.b_mean = np.array([10.0, 10.0, 10.0])
        mean = s._policy_forward(np.zeros(9))
        np.testing.assert_array_almost_equal(mean, [0.1, 0.1, 0.1])

    def test_sample_action_returns_correct_shapes(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        mean = np.zeros(3)
        action, log_prob = s._sample_action(mean)
        self.assertEqual(action.shape, (3,))
        self.assertIsInstance(log_prob, float)

    def test_log_prob_is_finite(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        mean = np.zeros(3)
        _, log_prob = s._sample_action(mean)
        self.assertTrue(np.isfinite(log_prob))

    def test_apply_action_zero_keeps_values(self):
        s = HyperparameterScheduler(0.001, 1.0, 2.0, seed=0)
        before = s.get_current_hyperparams()
        s._apply_action(np.array([0.0, 0.0, 0.0]))
        after = s.get_current_hyperparams()
        self.assertAlmostEqual(before['lr'], after['lr'])
        self.assertAlmostEqual(before['code_loss_scale'], after['code_loss_scale'])
        self.assertAlmostEqual(before['mle_loss_scale'], after['mle_loss_scale'])

    def test_apply_positive_action_increases_values(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        s._apply_action(np.array([0.1, 0.1, 0.1]))
        hp = s.get_current_hyperparams()
        self.assertGreater(hp['lr'], 0.001)
        self.assertGreater(hp['code_loss_scale'], 1.0)
        self.assertGreater(hp['mle_loss_scale'], 1.0)

    def test_apply_negative_action_decreases_values(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0)
        s._apply_action(np.array([-0.1, -0.1, -0.1]))
        hp = s.get_current_hyperparams()
        self.assertLess(hp['lr'], 0.001)
        self.assertLess(hp['code_loss_scale'], 1.0)
        self.assertLess(hp['mle_loss_scale'], 1.0)


class TestLongTrainingSimulation(unittest.TestCase):
    """Simulate a realistic training run and check stability."""

    def test_no_nan_or_inf_over_long_run(self):
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=42)
        s.set_total_epochs(200)
        rng = np.random.RandomState(123)
        for i in range(200):
            loss = 5.0 * np.exp(-i / 50.0) + rng.normal(0, 0.1)
            hp = s.step(
                train_loss_A=loss + rng.normal(0, 0.05),
                train_loss_B=loss + rng.normal(0, 0.05),
                code_loss=max(0, 1.0 - i * 0.005 + rng.normal(0, 0.01)),
                val_loss_A=loss + 0.2 + rng.normal(0, 0.05),
                val_loss_B=loss + 0.2 + rng.normal(0, 0.05))
            for v in hp.values():
                self.assertTrue(np.isfinite(v),
                                f"Non-finite value at step {i}: {hp}")

        # Policy weights should also be finite
        self.assertTrue(np.all(np.isfinite(s.W_mean)))
        self.assertTrue(np.all(np.isfinite(s.b_mean)))
        self.assertTrue(np.all(np.isfinite(s.log_std)))
        self.assertTrue(np.all(np.isfinite(s.W_value)))
        self.assertTrue(np.isfinite(s.b_value))

    def test_value_baseline_learns_direction(self):
        """With monotonically improving losses, baseline should learn positive values."""
        s = HyperparameterScheduler(0.001, 1.0, 1.0, seed=0, policy_lr=0.05)
        s.set_total_epochs(50)
        for i in range(50):
            s.step(10.0 - i * 0.15, 10.0 - i * 0.15,
                   2.0 - i * 0.03, 9.0 - i * 0.15, 9.0 - i * 0.15)
        # After consistent improvement, the value baseline should have
        # adapted (not all zeros)
        self.assertFalse(
            np.allclose(s.W_value, 0) and s.b_value == 0,
            "Value baseline should have learned something after 50 steps of improvement")


if __name__ == '__main__':
    unittest.main()
