"""Small NumPy D3QN used by the RL-SETS operator selector.

The project does not require PyTorch for its existing algorithms.  This
module therefore implements only the compact network needed by RL-SETS:
two ReLU layers, dueling value/advantage heads, Double-DQN targets, replay,
Adam updates and portable ``.npz`` checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ReplayBuffer:
    """Fixed-size replay memory with preallocated NumPy arrays."""

    def __init__(
        self,
        capacity,
        observation_size,
        action_count,
        rng,
        group_capacity=None,
    ):
        self.capacity = max(1, int(capacity))
        self.observation_size = int(observation_size)
        self.action_count = int(action_count)
        self.rng = rng
        self.group_capacity = (
            None if group_capacity is None else max(1, int(group_capacity))
        )
        self.observations = np.zeros(
            (self.capacity, self.observation_size), dtype=np.float32
        )
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones(
            (self.capacity, self.action_count), dtype=bool
        )
        self.group_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.position = 0
        self.size = 0

    def append(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
        next_mask,
        group_id=None,
    ):
        """Store one transition, optionally capping records from one map."""
        normalized_group = -1 if group_id is None else int(group_id)
        group_indices = np.empty(0, dtype=np.int64)
        if self.group_capacity is not None and normalized_group >= 0:
            group_indices = np.flatnonzero(
                self.group_ids[: self.size] == normalized_group
            )
        replacing_group_record = len(group_indices) >= (
            self.group_capacity or self.capacity + 1
        )
        index = (
            int(self.rng.choice(group_indices))
            if replacing_group_record
            else self.position
        )
        self.observations[index] = np.asarray(
            observation, dtype=np.float32
        )
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_observations[index] = np.asarray(
            next_observation, dtype=np.float32
        )
        self.dones[index] = float(bool(done))
        mask = np.asarray(next_mask, dtype=bool)
        if mask.shape != (self.action_count,):
            raise ValueError("next action mask has the wrong shape")
        if not np.any(mask):
            raise ValueError("next action mask must allow at least one action")
        self.next_masks[index] = mask
        self.group_ids[index] = normalized_group
        if not replacing_group_record:
            self.position = (index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch, balancing tagged training maps when enabled."""
        batch_size = min(int(batch_size), self.size)
        if batch_size <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        valid_groups = np.unique(self.group_ids[: self.size])
        valid_groups = valid_groups[valid_groups >= 0]
        if self.group_capacity is None or not len(valid_groups):
            indices = self.rng.choice(
                self.size, size=batch_size, replace=False
            )
        else:
            # Pick maps uniformly first, then a transition within each map.
            # This prevents long episodes from dominating a mixed-map batch.
            chosen_groups = self.rng.choice(
                valid_groups,
                size=batch_size,
                replace=len(valid_groups) < batch_size,
            )
            indices = np.asarray(
                [
                    self.rng.choice(
                        np.flatnonzero(self.group_ids[: self.size] == group)
                    )
                    for group in chosen_groups
                ],
                dtype=np.int64,
            )
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
            self.next_masks[indices],
        )

    def __len__(self):
        return self.size


class DuelingNetwork:
    """Fully connected dueling Q-network with explicit backpropagation."""

    PARAMETER_NAMES = (
        "weight1",
        "bias1",
        "weight2",
        "bias2",
        "value_weight",
        "value_bias",
        "advantage_weight",
        "advantage_bias",
    )

    def __init__(self, observation_size, action_count, hidden_size, rng):
        observation_size = int(observation_size)
        action_count = int(action_count)
        hidden_size = int(hidden_size)
        self.observation_size = observation_size
        self.action_count = action_count
        self.hidden_size = hidden_size

        def he(rows, columns):
            scale = np.sqrt(2.0 / max(1, rows))
            return rng.normal(0.0, scale, (rows, columns)).astype(np.float32)

        self.parameters = {
            "weight1": he(observation_size, hidden_size),
            "bias1": np.zeros(hidden_size, dtype=np.float32),
            "weight2": he(hidden_size, hidden_size),
            "bias2": np.zeros(hidden_size, dtype=np.float32),
            "value_weight": he(hidden_size, 1),
            "value_bias": np.zeros(1, dtype=np.float32),
            "advantage_weight": he(hidden_size, action_count),
            "advantage_bias": np.zeros(action_count, dtype=np.float32),
        }

    def predict(self, observations):
        """Return Q values for one observation or a batch."""
        values = np.asarray(observations, dtype=np.float32)
        one = values.ndim == 1
        if one:
            values = values[None, :]
        q_values, _ = self._forward(values)
        return q_values[0] if one else q_values

    def _forward(self, observations):
        p = self.parameters
        hidden1_pre = observations @ p["weight1"] + p["bias1"]
        hidden1 = np.maximum(hidden1_pre, 0.0)
        hidden2_pre = hidden1 @ p["weight2"] + p["bias2"]
        hidden2 = np.maximum(hidden2_pre, 0.0)
        value = hidden2 @ p["value_weight"] + p["value_bias"]
        advantage = (
            hidden2 @ p["advantage_weight"] + p["advantage_bias"]
        )
        q_values = value + advantage - np.mean(
            advantage, axis=1, keepdims=True
        )
        cache = (
            observations,
            hidden1_pre,
            hidden1,
            hidden2_pre,
            hidden2,
        )
        return q_values, cache

    def loss_and_gradients(self, observations, actions, targets):
        """Return mean Huber loss and gradients for selected Q values."""
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        q_values, cache = self._forward(observations)
        row_ids = np.arange(len(observations))
        errors = q_values[row_ids, actions] - targets
        absolute = np.abs(errors)
        losses = np.where(
            absolute <= 1.0,
            0.5 * errors**2,
            absolute - 0.5,
        )
        selected_gradient = np.where(
            absolute <= 1.0,
            errors,
            np.sign(errors),
        ) / max(1, len(observations))

        q_gradient = np.zeros_like(q_values)
        q_gradient[row_ids, actions] = selected_gradient
        value_gradient = np.sum(q_gradient, axis=1, keepdims=True)
        advantage_gradient = q_gradient - np.mean(
            q_gradient, axis=1, keepdims=True
        )

        observations, hidden1_pre, hidden1, hidden2_pre, hidden2 = cache
        p = self.parameters
        gradients = {
            "value_weight": hidden2.T @ value_gradient,
            "value_bias": np.sum(value_gradient, axis=0),
            "advantage_weight": hidden2.T @ advantage_gradient,
            "advantage_bias": np.sum(advantage_gradient, axis=0),
        }
        hidden2_gradient = (
            value_gradient @ p["value_weight"].T
            + advantage_gradient @ p["advantage_weight"].T
        )
        hidden2_gradient[hidden2_pre <= 0.0] = 0.0
        gradients["weight2"] = hidden1.T @ hidden2_gradient
        gradients["bias2"] = np.sum(hidden2_gradient, axis=0)
        hidden1_gradient = hidden2_gradient @ p["weight2"].T
        hidden1_gradient[hidden1_pre <= 0.0] = 0.0
        gradients["weight1"] = observations.T @ hidden1_gradient
        gradients["bias1"] = np.sum(hidden1_gradient, axis=0)
        return float(np.mean(losses)), gradients

    def copy_from(self, other):
        """Hard-copy all weights from another compatible network."""
        for name in self.PARAMETER_NAMES:
            self.parameters[name][...] = other.parameters[name]


class D3QNAgent:
    """Double DQN learner with a dueling network and replay memory."""

    def __init__(
        self,
        observation_size,
        action_count,
        *,
        hidden_size=64,
        replay_capacity=20000,
        replay_group_capacity=None,
        replay_warmup=256,
        batch_size=64,
        gamma=0.95,
        learning_rate=3e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        target_update_interval=500,
        gradient_clip=10.0,
        training=False,
        seed=None,
    ):
        self.observation_size = int(observation_size)
        self.action_count = int(action_count)
        self.hidden_size = int(hidden_size)
        self.replay_warmup = max(1, int(replay_warmup))
        self.batch_size = max(1, int(batch_size))
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.target_update_interval = max(1, int(target_update_interval))
        self.gradient_clip = float(gradient_clip)
        self.training = bool(training)
        self.rng = np.random.default_rng(seed)

        self.online = DuelingNetwork(
            self.observation_size,
            self.action_count,
            self.hidden_size,
            self.rng,
        )
        self.target = DuelingNetwork(
            self.observation_size,
            self.action_count,
            self.hidden_size,
            self.rng,
        )
        self.target.copy_from(self.online)
        self.replay = ReplayBuffer(
            replay_capacity,
            self.observation_size,
            self.action_count,
            self.rng,
            group_capacity=replay_group_capacity,
        )
        self.replay_group = None
        self.environment_steps = 0
        self.training_steps = 0
        # 記錄 reward 等訓練契約，避免同形狀網路混用不同學習目標。
        self.checkpoint_metadata = {}
        self.last_loss = None
        self._adam_step = 0
        self._adam_mean = {
            name: np.zeros_like(value)
            for name, value in self.online.parameters.items()
        }
        self._adam_variance = {
            name: np.zeros_like(value)
            for name, value in self.online.parameters.items()
        }

    @property
    def epsilon(self):
        """Current epsilon; frozen evaluation always reports zero."""
        if not self.training:
            return 0.0
        return max(
            self.epsilon_end,
            self.epsilon_start * self.epsilon_decay**self.environment_steps,
        )

    def select_actions(self, observations, masks=None):
        """Select one valid epsilon-greedy action per observation."""
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations[None, :]
        if observations.shape[1:] != (self.observation_size,):
            raise ValueError("observation batch has the wrong shape")
        if masks is None:
            masks = np.ones(
                (len(observations), self.action_count), dtype=bool
            )
        masks = np.asarray(masks, dtype=bool)
        if masks.shape != (len(observations), self.action_count):
            raise ValueError("action mask batch has the wrong shape")
        if np.any(~np.any(masks, axis=1)):
            raise ValueError("every observation must allow one action")

        q_values = self.online.predict(observations)
        actions = np.empty(len(observations), dtype=int)
        epsilon = self.epsilon
        for row in range(len(observations)):
            valid = np.flatnonzero(masks[row])
            if self.training and self.rng.random() < epsilon:
                actions[row] = int(self.rng.choice(valid))
                continue
            valid_q = q_values[row, valid]
            best = valid[np.flatnonzero(valid_q == np.max(valid_q))]
            actions[row] = int(self.rng.choice(best))
        if self.training:
            self.environment_steps += len(actions)
        return actions

    def remember(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
        next_mask,
    ):
        """Add one transition only while the agent is training."""
        if self.training:
            self.replay.append(
                observation,
                action,
                reward,
                next_observation,
                done,
                next_mask,
                self.replay_group,
            )

    def set_replay_group(self, group_id):
        """Tag subsequent transitions with a stable training-map id."""
        self.replay_group = None if group_id is None else int(group_id)

    def learn(self):
        """Perform one Double-DQN update, or return ``None`` before warmup."""
        if not self.training or len(self.replay) < self.replay_warmup:
            return None
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            next_masks,
        ) = self.replay.sample(self.batch_size)

        online_next = self.online.predict(next_observations)
        masked_online_next = np.where(next_masks, online_next, -np.inf)
        next_actions = np.argmax(masked_online_next, axis=1)
        target_next = self.target.predict(next_observations)
        next_values = target_next[np.arange(len(next_actions)), next_actions]
        targets = rewards + self.gamma * (1.0 - dones) * next_values

        loss, gradients = self.online.loss_and_gradients(
            observations, actions, targets
        )
        self._apply_adam(gradients)
        self.training_steps += 1
        self.last_loss = loss
        if self.training_steps % self.target_update_interval == 0:
            self.target.copy_from(self.online)
        return loss

    def _apply_adam(self, gradients):
        """Apply one globally clipped Adam update."""
        squared_norm = sum(
            float(np.sum(np.asarray(gradient, dtype=float) ** 2))
            for gradient in gradients.values()
        )
        norm = np.sqrt(squared_norm)
        scale = (
            1.0
            if norm <= self.gradient_clip or norm == 0.0
            else self.gradient_clip / norm
        )
        self._adam_step += 1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for name, gradient in gradients.items():
            gradient = np.asarray(gradient, dtype=np.float32) * scale
            mean = self._adam_mean[name]
            variance = self._adam_variance[name]
            mean *= beta1
            mean += (1.0 - beta1) * gradient
            variance *= beta2
            variance += (1.0 - beta2) * gradient**2
            mean_hat = mean / (1.0 - beta1**self._adam_step)
            variance_hat = variance / (1.0 - beta2**self._adam_step)
            self.online.parameters[name] -= (
                self.learning_rate
                * mean_hat
                / (np.sqrt(variance_hat) + epsilon)
            )

    def save(self, path):
        """Save model, target weights and optimizer state to one NPZ file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "observation_size": np.asarray(self.observation_size),
            "action_count": np.asarray(self.action_count),
            "hidden_size": np.asarray(self.hidden_size),
            "environment_steps": np.asarray(self.environment_steps),
            "training_steps": np.asarray(self.training_steps),
            "adam_step": np.asarray(self._adam_step),
            "metadata_json": np.asarray(json.dumps(self.checkpoint_metadata)),
            "rng_state_json": np.asarray(json.dumps(self.rng.bit_generator.state)),
            "replay_capacity": np.asarray(self.replay.capacity),
            "replay_group_capacity": np.asarray(
                -1
                if self.replay.group_capacity is None
                else self.replay.group_capacity
            ),
            "replay_size": np.asarray(self.replay.size),
            "replay_position": np.asarray(self.replay.position),
            "replay_observations": self.replay.observations[: self.replay.size],
            "replay_actions": self.replay.actions[: self.replay.size],
            "replay_rewards": self.replay.rewards[: self.replay.size],
            "replay_next_observations": self.replay.next_observations[
                : self.replay.size
            ],
            "replay_dones": self.replay.dones[: self.replay.size],
            "replay_next_masks": self.replay.next_masks[: self.replay.size],
            "replay_group_ids": self.replay.group_ids[: self.replay.size],
        }
        for name in DuelingNetwork.PARAMETER_NAMES:
            values[f"online_{name}"] = self.online.parameters[name]
            values[f"target_{name}"] = self.target.parameters[name]
            values[f"adam_mean_{name}"] = self._adam_mean[name]
            values[f"adam_variance_{name}"] = self._adam_variance[name]
        with path.open("wb") as model_file:
            np.savez_compressed(model_file, **values)
        return path

    def load(self, path):
        """Load a compatible checkpoint without enabling pickle."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as values:
            expected = (
                self.observation_size,
                self.action_count,
                self.hidden_size,
            )
            actual = (
                int(values["observation_size"]),
                int(values["action_count"]),
                int(values["hidden_size"]),
            )
            if actual != expected:
                raise ValueError(
                    f"D3QN checkpoint shape {actual} does not match {expected}"
                )
            for name in DuelingNetwork.PARAMETER_NAMES:
                self.online.parameters[name][...] = values[f"online_{name}"]
                self.target.parameters[name][...] = values[f"target_{name}"]
                mean_key = f"adam_mean_{name}"
                variance_key = f"adam_variance_{name}"
                if mean_key in values:
                    self._adam_mean[name][...] = values[mean_key]
                if variance_key in values:
                    self._adam_variance[name][...] = values[variance_key]
            self.environment_steps = int(values["environment_steps"])
            self.training_steps = int(values["training_steps"])
            self._adam_step = int(values.get("adam_step", 0))
            # 舊模型沒有 metadata，仍可供凍結推論；續訓由演算法檢查版本。
            self.checkpoint_metadata = (
                json.loads(str(values["metadata_json"]))
                if "metadata_json" in values else {}
            )
            if "rng_state_json" in values:
                self.rng.bit_generator.state = json.loads(
                    str(values["rng_state_json"])
                )
            if self.training and "replay_size" in values:
                replay_size = int(values["replay_size"])
                if replay_size > self.replay.capacity:
                    raise ValueError(
                        "checkpoint replay is larger than configured capacity"
                    )
                saved_group_capacity = int(
                    values.get("replay_group_capacity", -1)
                )
                configured_group_capacity = (
                    -1
                    if self.replay.group_capacity is None
                    else self.replay.group_capacity
                )
                if saved_group_capacity != configured_group_capacity:
                    raise ValueError(
                        "checkpoint replay group capacity does not match"
                    )
                self.replay.observations[:replay_size] = values[
                    "replay_observations"
                ]
                self.replay.actions[:replay_size] = values["replay_actions"]
                self.replay.rewards[:replay_size] = values["replay_rewards"]
                self.replay.next_observations[:replay_size] = values[
                    "replay_next_observations"
                ]
                self.replay.dones[:replay_size] = values["replay_dones"]
                self.replay.next_masks[:replay_size] = values[
                    "replay_next_masks"
                ]
                self.replay.group_ids[:replay_size] = values[
                    "replay_group_ids"
                ]
                self.replay.size = replay_size
                self.replay.position = int(values["replay_position"])
        return path


__all__ = ["D3QNAgent", "DuelingNetwork", "ReplayBuffer"]
