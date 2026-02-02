"""
RL-based hyperparameter scheduler for pix2pix-flow training.

Uses REINFORCE with a learned baseline to dynamically adjust:
  - learning rate (lr)
  - code_loss_scale
  - mle_loss_scale

The agent observes recent training metrics and outputs multiplicative
adjustments to each hyperparameter at the end of every epoch.

The reward signal is the negative validation loss improvement,
encouraging the agent to find schedules that steadily reduce loss.
"""

import numpy as np
import os
import json


class HyperparameterScheduler:
    """RL agent that tunes hyperparameters during training.

    State vector (per epoch):
        [mean_train_loss_A, mean_train_loss_B, mean_code_loss,
         val_loss_A, val_loss_B,
         current_lr_ratio, current_code_scale, current_mle_scale,
         epoch_progress]

    Actions (continuous, Gaussian policy):
        [lr_log_multiplier, code_scale_log_multiplier, mle_scale_log_multiplier]

    Each action is a log-multiplier clipped to [-action_bound, +action_bound],
    so the actual multiplier applied is exp(action), keeping values positive.
    """

    STATE_DIM = 9
    ACTION_DIM = 3

    def __init__(self, base_lr, base_code_loss_scale, base_mle_loss_scale,
                 lr_range=(1e-6, 1e-2),
                 code_scale_range=(0.01, 100.0),
                 mle_scale_range=(0.01, 100.0),
                 policy_lr=0.01,
                 action_bound=0.3,
                 discount=0.95,
                 entropy_coeff=0.01,
                 seed=None):
        """
        Args:
            base_lr: Initial learning rate from hps.
            base_code_loss_scale: Initial code_loss_scale from hps.
            base_mle_loss_scale: Initial mle_loss_scale from hps.
            lr_range: (min, max) clamp for learning rate.
            code_scale_range: (min, max) clamp for code_loss_scale.
            mle_scale_range: (min, max) clamp for mle_loss_scale.
            policy_lr: Learning rate for the policy network.
            action_bound: Max absolute value of log-multiplier actions.
            discount: Discount factor for returns.
            entropy_coeff: Entropy bonus coefficient for exploration.
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()

        self.base_lr = base_lr
        self.base_code_loss_scale = base_code_loss_scale
        self.base_mle_loss_scale = base_mle_loss_scale

        # Current hyperparameter values
        self.current_lr = base_lr
        self.current_code_loss_scale = base_code_loss_scale
        self.current_mle_loss_scale = base_mle_loss_scale

        # Clamp ranges
        self.lr_range = lr_range
        self.code_scale_range = code_scale_range
        self.mle_scale_range = mle_scale_range

        self.action_bound = action_bound
        self.discount = discount
        self.entropy_coeff = entropy_coeff

        # Simple linear policy: mean = W @ state + b
        self._policy_lr = policy_lr
        self.W_mean = np.zeros((self.ACTION_DIM, self.STATE_DIM), dtype=np.float64)
        self.b_mean = np.zeros(self.ACTION_DIM, dtype=np.float64)

        # Log standard deviation (learnable, per action dimension)
        self.log_std = np.full(self.ACTION_DIM, -1.0, dtype=np.float64)

        # Value baseline: linear V(s) = w @ state + b
        self.W_value = np.zeros(self.STATE_DIM, dtype=np.float64)
        self.b_value = 0.0
        self._value_lr = policy_lr * 2.0

        # Episode buffer
        self._states = []
        self._actions = []
        self._rewards = []
        self._log_probs = []

        # Tracking
        self._prev_val_loss = None
        self._epoch_count = 0
        self._total_epochs = 1  # Set via set_total_epochs

        # Running stats for state normalization
        self._state_mean = np.zeros(self.STATE_DIM, dtype=np.float64)
        self._state_var = np.ones(self.STATE_DIM, dtype=np.float64)
        self._state_count = 0

    def set_total_epochs(self, total_epochs):
        """Set total training epochs for progress feature."""
        self._total_epochs = max(total_epochs, 1)

    def _normalize_state(self, state):
        """Incrementally normalize states using Welford's online algorithm."""
        state = np.asarray(state, dtype=np.float64)
        self._state_count += 1
        n = self._state_count
        delta = state - self._state_mean
        self._state_mean += delta / n
        delta2 = state - self._state_mean
        self._state_var += (delta * delta2 - self._state_var) / n
        std = np.sqrt(np.maximum(self._state_var, 1e-8))
        return (state - self._state_mean) / std

    def _build_state(self, train_loss_A, train_loss_B, code_loss,
                     val_loss_A, val_loss_B):
        """Construct the state vector from training metrics."""
        epoch_progress = self._epoch_count / self._total_epochs
        lr_ratio = self.current_lr / self.base_lr
        return np.array([
            train_loss_A,
            train_loss_B,
            code_loss,
            val_loss_A,
            val_loss_B,
            lr_ratio,
            self.current_code_loss_scale,
            self.current_mle_loss_scale,
            epoch_progress,
        ], dtype=np.float64)

    def _policy_forward(self, state_norm):
        """Compute action mean from linear policy."""
        mean = self.W_mean @ state_norm + self.b_mean
        return np.clip(mean, -self.action_bound, self.action_bound)

    def _value_forward(self, state_norm):
        """Compute baseline value estimate."""
        return float(self.W_value @ state_norm + self.b_value)

    def _sample_action(self, mean):
        """Sample action from Gaussian policy."""
        std = np.exp(self.log_std)
        noise = self._rng.randn(self.ACTION_DIM)
        action = mean + std * noise
        action = np.clip(action, -self.action_bound, self.action_bound)

        # Log probability of action under Gaussian
        log_prob = -0.5 * np.sum(((action - mean) / std) ** 2) \
                   - 0.5 * self.ACTION_DIM * np.log(2 * np.pi) \
                   - np.sum(self.log_std)
        return action, log_prob

    def _apply_action(self, action):
        """Apply log-multiplier action to current hyperparameters."""
        multipliers = np.exp(action)

        self.current_lr = np.clip(
            self.current_lr * multipliers[0],
            *self.lr_range)
        self.current_code_loss_scale = np.clip(
            self.current_code_loss_scale * multipliers[1],
            *self.code_scale_range)
        self.current_mle_loss_scale = np.clip(
            self.current_mle_loss_scale * multipliers[2],
            *self.mle_scale_range)

    def step(self, train_loss_A, train_loss_B, code_loss,
             val_loss_A, val_loss_B):
        """Called at the end of each epoch after validation.

        Computes reward from validation improvement, selects new
        hyperparameter values, and stores the transition.

        Args:
            train_loss_A: Mean training loss for model A this epoch.
            train_loss_B: Mean training loss for model B this epoch.
            code_loss: Mean code alignment loss this epoch.
            val_loss_A: Validation loss for model A.
            val_loss_B: Validation loss for model B.

        Returns:
            dict with updated 'lr', 'code_loss_scale', 'mle_loss_scale'.
        """
        self._epoch_count += 1

        # Build state
        state = self._build_state(
            train_loss_A, train_loss_B, code_loss, val_loss_A, val_loss_B)
        state_norm = self._normalize_state(state)

        # Compute reward: negative combined validation loss improvement
        current_val = val_loss_A + val_loss_B
        if self._prev_val_loss is not None:
            reward = self._prev_val_loss - current_val  # positive if improved
        else:
            reward = 0.0
        self._prev_val_loss = current_val

        # Store reward for previous transition
        if len(self._actions) > 0:
            self._rewards.append(reward)

        # Select action
        mean = self._policy_forward(state_norm)
        action, log_prob = self._sample_action(mean)

        # Store transition
        self._states.append(state_norm)
        self._actions.append(action)
        self._log_probs.append(log_prob)

        # Apply action
        self._apply_action(action)

        # Update policy periodically (every episode_len steps)
        if len(self._rewards) >= 10:
            self._update_policy()

        return {
            'lr': float(self.current_lr),
            'code_loss_scale': float(self.current_code_loss_scale),
            'mle_loss_scale': float(self.current_mle_loss_scale),
        }

    def _compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = np.zeros(len(rewards), dtype=np.float64)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.discount * running
            returns[t] = running
        return returns

    def _update_policy(self):
        """REINFORCE update with baseline subtraction."""
        rewards = np.array(self._rewards, dtype=np.float64)
        # Match lengths: we have one fewer reward than states/actions
        n = len(rewards)
        states = np.array(self._states[:n], dtype=np.float64)
        actions = np.array(self._actions[:n], dtype=np.float64)
        log_probs = np.array(self._log_probs[:n], dtype=np.float64)

        if n == 0:
            return

        returns = self._compute_returns(rewards)

        # Normalize returns
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute baselines
        baselines = np.array([self._value_forward(s) for s in states])
        advantages = returns - baselines

        # Policy gradient: d/d_theta log pi(a|s) * advantage
        std = np.exp(self.log_std)
        for t in range(n):
            s = states[t]
            a = actions[t]
            adv = advantages[t]
            mean = self._policy_forward(s)

            # Gradient of log pi w.r.t. mean
            d_mean = (a - mean) / (std ** 2)  # (ACTION_DIM,)

            # Gradient of mean w.r.t. W_mean and b_mean (linear policy)
            # mean = W @ s + b, so d_mean/d_W = d_mean outer s, d_mean/d_b = d_mean
            grad_W = np.outer(d_mean, s) * adv
            grad_b = d_mean * adv

            # Gradient of log pi w.r.t. log_std
            grad_log_std = ((a - mean) ** 2 / (std ** 2) - 1.0) * adv
            # Add entropy bonus gradient: d/d_log_std (log_std) = 1
            grad_log_std += self.entropy_coeff

            self.W_mean += self._policy_lr * grad_W
            self.b_mean += self._policy_lr * grad_b
            self.log_std += self._policy_lr * 0.1 * grad_log_std

        # Clamp log_std to reasonable range
        self.log_std = np.clip(self.log_std, -3.0, 1.0)

        # Update value baseline via MSE gradient
        for t in range(n):
            s = states[t]
            v = self._value_forward(s)
            td_error = returns[t] - v
            self.W_value += self._value_lr * td_error * s
            self.b_value += self._value_lr * td_error

        # Clear buffers but keep the last state/action for continuity
        self._states = self._states[n:]
        self._actions = self._actions[n:]
        self._log_probs = self._log_probs[n:]
        self._rewards = []

    def get_current_hyperparams(self):
        """Return current hyperparameter values."""
        return {
            'lr': float(self.current_lr),
            'code_loss_scale': float(self.current_code_loss_scale),
            'mle_loss_scale': float(self.current_mle_loss_scale),
        }

    def save(self, path):
        """Save agent state to JSON."""
        state = {
            'W_mean': self.W_mean.tolist(),
            'b_mean': self.b_mean.tolist(),
            'log_std': self.log_std.tolist(),
            'W_value': self.W_value.tolist(),
            'b_value': self.b_value,
            'current_lr': self.current_lr,
            'current_code_loss_scale': self.current_code_loss_scale,
            'current_mle_loss_scale': self.current_mle_loss_scale,
            'state_mean': self._state_mean.tolist(),
            'state_var': self._state_var.tolist(),
            'state_count': self._state_count,
            'prev_val_loss': self._prev_val_loss,
            'epoch_count': self._epoch_count,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, path):
        """Load agent state from JSON."""
        with open(path, 'r') as f:
            state = json.load(f)
        self.W_mean = np.array(state['W_mean'], dtype=np.float64)
        self.b_mean = np.array(state['b_mean'], dtype=np.float64)
        self.log_std = np.array(state['log_std'], dtype=np.float64)
        self.W_value = np.array(state['W_value'], dtype=np.float64)
        self.b_value = state['b_value']
        self.current_lr = state['current_lr']
        self.current_code_loss_scale = state['current_code_loss_scale']
        self.current_mle_loss_scale = state['current_mle_loss_scale']
        self._state_mean = np.array(state['state_mean'], dtype=np.float64)
        self._state_var = np.array(state['state_var'], dtype=np.float64)
        self._state_count = state['state_count']
        self._prev_val_loss = state['prev_val_loss']
        self._epoch_count = state['epoch_count']
