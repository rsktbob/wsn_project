"""Ring 家族的逐組區段選擇：每組 searcher-good 交易各自選一個區段。

原本的 Ring_SETS 讓每個 searcher 每回合選一個區段，它的 w 個 goods 都在
同一區交配。這裡改成每組交易各自決定：決策前先在每個區段用原本的 80/20
規則抽好一個範圍，再依「這個範圍裡 searcher 和 good 差在哪」選一個區段。
範圍大小、交配、突變、菁英更新都和 Ring_SETS 相同。

* 有效差異：兩者在這顆 sensor 的基因不同，而且任一方解碼後有開這顆 sensor。
  兩邊都沒開的 sensor，改基因幾乎不會改變解（平坦比例分析）。
* 回饋：0.5 × [child1 比 searcher 好] + 0.5 × [child2 比 good 好]。

``LinUCBSegmentPolicy``：共用一個線性模型，模型跨 lifetime 段延續。
``D3QNSegmentPolicy``：每個區段共用權重的 dueling 網路，事先訓練後凍結使用。
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

from Algorithm.se.d3qn import D3QNAgent, DuelingNetwork
from Algorithm.se.market_components import (
    accept_good_children,
    accept_searcher_children,
)
from State.Encoding import swap_segment
from State.SensorEncoding import DEFAULT_SENSOR_ENCODING, SensorEncoding
from State.State import State

IMPROVEMENT_EPSILON = 1e-12


# RL 策略是凍結的 .npz，只在訓練當下那組 fitness／routing／decode 實作下有意義。
# 把 problem 用到的實作組合雜湊起來，載入 checkpoint 時比對，避免環境改過卻默默
# 跑出無意義的結果。（原本在 RL_SETS.py，2026-10-04 該系列移到 legacy/；內容不變，舊 checkpoint 仍相容。）
def environment_contract(problem):
    """Record actual implementations, including changes within a version."""
    implementations = {
        "fitness": type(problem.fitness_service),
        "routing": type(problem.routing_service),
        "decode": SensorEncoding,
        "problem": type(problem),
        "state": State,
        "energy": type(problem.energy_service),
        "coverage": type(problem.coverage_service),
        "geometry": type(problem.G),
    }
    # Hash inherited service methods too, not only each leaf class file.
    sources = {
        f"{base.__module__}.{base.__name__}":
            hashlib.sha256(Path(inspect.getfile(base)).read_bytes()).hexdigest()
        for implementation in implementations.values()
        for base in implementation.__mro__ if base is not object
    }
    return {
        "sensing_mode": problem.sensing_mode,
        "sensor_encoding": DEFAULT_SENSOR_ENCODING,
        "implementations": {
            key: f"{value.__module__}.{value.__name__}"
            for key, value in implementations.items()
        },
        "source_sha256": sources,
    }


class PairSegmentMarket:
    """Ring_SETS round where every searcher-good pair picks its own segment.

    Subclasses provide ``_choose_segments(context, pairs)`` and
    ``_observe_round(context, pairs, choices, rewards, done)``.
    """

    # 只有 D3QN 需要瓶頸 sensor 與電量特徵；LinUCB 不必多算成本。
    NEEDS_ENERGY_FEATURES = False
    STAGNATION_SCALE = 20

    def initialize_market(self, problem, initial_state=None):
        # 解碼只取決於染色體與目前電量；電量只在段與段之間改變，所以每段清空。
        self._decode_cache = {}
        super().initialize_market(problem, initial_state)
        self.searcher_stagnation = np.zeros(self.n, dtype=int)
        self.max_region_length = max(b - a for a, b in self.region_sensor_bounds)
        self.round_rewards = []

    # ---- 每回合共用的解碼資訊 ---------------------------------------------
    def _decoded(self, problem, candidate, need_cost=False):
        """Active mask (and per-sensor cost) of a candidate, cached by chromosome.

        Most searchers and goods are unchanged between rounds, so only new
        chromosomes are decoded.  The values are identical to decoding again.
        """
        key = np.asarray(candidate.code).tobytes()
        info = self._decode_cache.get(key)
        if info is None or (need_cost and "cost" not in info):
            state = candidate.decode(problem)
            info = {"active": np.asarray(state.levels) > 0}
            if need_cost:
                radii = np.asarray(problem.state_radius(state), dtype=float)
                info["cost"] = np.asarray(
                    problem.calculate_total_cost(state, sensing_radii=radii),
                    dtype=float,
                )[: problem.SENSOR_NUMBER]
            self._decode_cache[key] = info
        return info

    def _round_context(self, problem):
        context = {
            "progress": min(1.0, self.evatime / max(1, self.evaluation_limit)),
            "searcher_active": [],
            "goods_active": [
                self._decoded(problem, good)["active"] for good in self.goods
            ],
        }
        energy = np.maximum(np.asarray(problem.energy, dtype=float), 0.0)
        energy_ratio = energy / max(float(problem.initial_energy), 1e-15)
        context["energy_ratio"] = energy_ratio
        bottlenecks = []
        for searcher in self.searchers:
            info = self._decoded(problem, searcher, need_cost=self.NEEDS_ENERGY_FEATURES)
            context["searcher_active"].append(info["active"])
            if not self.NEEDS_ENERGY_FEATURES:
                continue
            cost = info["cost"]
            spending = np.isfinite(cost) & (cost > 1e-15)
            if np.any(spending):
                ids = np.flatnonzero(spending)
                bottlenecks.append(int(ids[np.argmin(energy[ids] / cost[ids])]))
            else:
                bottlenecks.append(-1)
        context["searcher_bottleneck"] = bottlenecks
        return context

    # ---- 每組交易的候選範圍與統計 -----------------------------------------
    def _pair_segments(self, context, searcher_id, good_id):
        """Draw one span per segment and summarize what each span exchanges."""
        code_s = np.asarray(self.searchers[searcher_id].code)
        code_g = np.asarray(self.goods[good_id].code)
        differ = (code_s[0::2] != code_g[0::2]) | (code_s[1::2] != code_g[1::2])
        active_s = context["searcher_active"][searcher_id]
        either_on = active_s | context["goods_active"][good_id]
        effective = differ & either_on
        inactive = differ & ~either_on
        spans, stats = [], []
        for region in range(self.h):
            span = self.select_sensor_span(region)
            _, left, right = span
            length = right - left
            stat = {
                "effective": int(effective[left:right].sum()),
                "inactive": int(inactive[left:right].sum()),
                "length": length / self.max_region_length,
                "active_ratio": float(active_s[left:right].sum()) / max(length, 1),
            }
            if self.NEEDS_ENERGY_FEATURES:
                bottleneck = context["searcher_bottleneck"][searcher_id]
                on = active_s[left:right]
                stat["bottleneck"] = float(left <= bottleneck < right)
                stat["energy"] = (
                    float(np.mean(context["energy_ratio"][left:right][on]))
                    if np.any(on) else 0.0
                )
            spans.append(span)
            stats.append(stat)
        return spans, stats

    # ---- 一回合 -------------------------------------------------------------
    def vision_search(self, problem):
        goods_fitness_before = self.goods_fitness.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        self._best_before = float(self.fitness)
        context = self._round_context(problem)
        pairs = self._build_pairs(context)
        self._last_pairs = pairs
        choices = self._choose_segments(context, pairs)

        children = [[] for _ in range(self.n)]
        for (s_id, g_id, spans, _), region in zip(pairs, choices):
            span = spans[int(region)]
            child1, child2 = swap_segment(
                self.searchers[s_id], self.goods[g_id], span[1] * 2, span[2] * 2
            )
            for child in (child1, child2):
                if self.random.random() < self.mutation_rate:
                    self.mutate_candidate(problem, child, sensor_span=span)
            children[s_id] += [child1, child2]

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]
        rewards = self._pair_rewards(
            pairs, child1_fitness, child2_fitness,
            searcher_fitness_before, goods_fitness_before,
        )
        self.round_rewards.append(float(np.mean(rewards)))

        accept_searcher_children(self, children, child1_fitness, searcher_fitness_before)
        accept_good_children(
            self.goods, self.goods_fitness, goods_fitness_before, children,
            child2_fitness, range(self.n), self.IMPROVEMENT_TOLERANCE,
        )
        improved = self.searcher_fitness > searcher_fitness_before + IMPROVEMENT_EPSILON
        self.searcher_stagnation = np.where(improved, 0, self.searcher_stagnation + 1)

        done = (
            self.evatime >= self.evaluation_limit
            or self.iterations_completed + 1 >= self.max_iterations
        )
        self._observe_round(context, pairs, choices, rewards, done)

    def _build_pairs(self, context):
        """One (searcher, good, spans, stats) entry per searcher-good pair."""
        return [
            (s_id, g_id, *self._pair_segments(context, s_id, g_id))
            for s_id in range(self.n)
            for g_id in range(self.w)
        ]

    def _pair_rewards(self, pairs, child1_fitness, child2_fitness,
                      searcher_before, goods_before):
        """0.5 × [child1 improves its searcher] + 0.5 × [child2 improves its good]."""
        return np.array([
            0.5 * float(child1_fitness[s_id, g_id] > searcher_before[s_id] + IMPROVEMENT_EPSILON)
            + 0.5 * float(child2_fitness[s_id, g_id] > goods_before[g_id] + IMPROVEMENT_EPSILON)
            for s_id, g_id, _, _ in pairs
        ])

    def update_search_memory(self, problem, evaluation_start):
        """No region memory; only record this round's fitness history."""
        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness


class LinUCBSegmentPolicy(PairSegmentMarket):
    """Shared-parameter LinUCB over the h candidate spans of each pair.

    Features (one row per segment): constant, effective differences in
    {1-2, 3-4, >=5} (0 is the reference), inactive differences / 5, span
    length, searcher active ratio, search progress and progress × {1-2}.
    The model persists across optimizer calls of one lifetime run.
    """

    ALPHA = 0.3
    RIDGE = 1.0
    FEATURE_NAMES = (
        "constant", "effective_1_2", "effective_3_4", "effective_5_plus",
        "inactive_differences", "span_length", "searcher_active_ratio",
        "progress", "progress_x_effective_1_2",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size = len(self.FEATURE_NAMES)
        self.linucb_A = self.RIDGE * np.eye(size)
        self.linucb_b = np.zeros(size)

    def _linucb_features(self, stat, progress):
        e = stat["effective"]
        small = 1.0 if 1 <= e <= 2 else 0.0
        return np.array([
            1.0, small, 1.0 if 3 <= e <= 4 else 0.0, 1.0 if e >= 5 else 0.0,
            min(stat["inactive"], 5) / 5.0, stat["length"], stat["active_ratio"],
            progress, progress * small,
        ])

    def _choose_segments(self, context, pairs):
        A_inv = np.linalg.inv(self.linucb_A)
        theta = A_inv @ self.linucb_b
        self._chosen_features = []
        choices = []
        for _, _, _, stats in pairs:
            X = np.array([self._linucb_features(s, context["progress"]) for s in stats])
            scores = X @ theta + self.ALPHA * np.sqrt(
                np.einsum("ij,jk,ik->i", X, A_inv, X)
            )
            best = np.flatnonzero(scores >= scores.max() - 1e-12)
            pick = int(best[self.random.randrange(len(best))])
            choices.append(pick)
            self._chosen_features.append(X[pick])
        return choices

    def _observe_round(self, context, pairs, choices, rewards, done):
        for x, reward in zip(self._chosen_features, rewards):
            self.linucb_A += np.outer(x, x)
            self.linucb_b += reward * x


class SharedSegmentDuelingNetwork:
    """Dueling Q-network whose advantage head is shared across segments.

    Each segment row ``[segment features, global features]`` passes through
    the same two ReLU layers.  A(s, k) is a linear head on segment k's hidden
    vector; V(s) is a linear head on the mean hidden vector.  Q = V + A − mean A.
    Parameter names match :class:`DuelingNetwork`, so the existing agent's
    Adam, target copy and ``.npz`` save/load work unchanged.
    """

    PARAMETER_NAMES = DuelingNetwork.PARAMETER_NAMES

    def __init__(self, segment_count, segment_features, global_features, hidden_size, rng):
        self.segment_count = int(segment_count)
        self.segment_features = int(segment_features)
        self.global_features = int(global_features)
        self.observation_size = self.segment_count * self.segment_features + self.global_features
        self.action_count = self.segment_count
        self.hidden_size = int(hidden_size)
        row_size = self.segment_features + self.global_features

        def he(rows, columns):
            return rng.normal(0.0, np.sqrt(2.0 / max(1, rows)), (rows, columns)).astype(np.float32)

        self.parameters = {
            "weight1": he(row_size, self.hidden_size),
            "bias1": np.zeros(self.hidden_size, dtype=np.float32),
            "weight2": he(self.hidden_size, self.hidden_size),
            "bias2": np.zeros(self.hidden_size, dtype=np.float32),
            "value_weight": he(self.hidden_size, 1),
            "value_bias": np.zeros(1, dtype=np.float32),
            "advantage_weight": he(self.hidden_size, 1),
            "advantage_bias": np.zeros(1, dtype=np.float32),
        }

    def _rows(self, observations):
        batch = len(observations)
        split = self.segment_count * self.segment_features
        segments = observations[:, :split].reshape(batch, self.segment_count, self.segment_features)
        shared = np.repeat(observations[:, None, split:], self.segment_count, axis=1)
        return np.concatenate((segments, shared), axis=2).reshape(batch * self.segment_count, -1)

    def _forward(self, observations):
        p = self.parameters
        batch = len(observations)
        rows = self._rows(observations)
        hidden1_pre = rows @ p["weight1"] + p["bias1"]
        hidden1 = np.maximum(hidden1_pre, 0.0)
        hidden2_pre = hidden1 @ p["weight2"] + p["bias2"]
        hidden2 = np.maximum(hidden2_pre, 0.0)
        per_segment = hidden2.reshape(batch, self.segment_count, self.hidden_size)
        advantage = (hidden2 @ p["advantage_weight"] + p["advantage_bias"]).reshape(batch, self.segment_count)
        pooled = per_segment.mean(axis=1)
        value = pooled @ p["value_weight"] + p["value_bias"]
        q_values = value + advantage - advantage.mean(axis=1, keepdims=True)
        return q_values, (rows, hidden1_pre, hidden1, hidden2_pre, hidden2, pooled)

    def predict(self, observations):
        values = np.asarray(observations, dtype=np.float32)
        one = values.ndim == 1
        if one:
            values = values[None, :]
        q_values, _ = self._forward(values)
        return q_values[0] if one else q_values

    def loss_and_gradients(self, observations, actions, targets):
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        q_values, cache = self._forward(observations)
        batch = len(observations)
        row_ids = np.arange(batch)
        errors = q_values[row_ids, actions] - targets
        absolute = np.abs(errors)
        losses = np.where(absolute <= 1.0, 0.5 * errors**2, absolute - 0.5)
        selected = np.where(absolute <= 1.0, errors, np.sign(errors)) / max(1, batch)
        q_gradient = np.zeros_like(q_values)
        q_gradient[row_ids, actions] = selected
        value_gradient = q_gradient.sum(axis=1, keepdims=True)
        advantage_gradient = q_gradient - q_gradient.mean(axis=1, keepdims=True)

        rows, hidden1_pre, hidden1, hidden2_pre, hidden2, pooled = cache
        p = self.parameters
        flat_advantage = advantage_gradient.reshape(-1, 1)
        gradients = {
            "value_weight": pooled.T @ value_gradient,
            "value_bias": value_gradient.sum(axis=0),
            "advantage_weight": hidden2.T @ flat_advantage,
            "advantage_bias": flat_advantage.sum(axis=0),
        }
        pooled_gradient = value_gradient @ p["value_weight"].T
        hidden2_gradient = (
            flat_advantage @ p["advantage_weight"].T
            + np.repeat(pooled_gradient / self.segment_count, self.segment_count, axis=0)
        )
        hidden2_gradient[hidden2_pre <= 0.0] = 0.0
        gradients["weight2"] = hidden1.T @ hidden2_gradient
        gradients["bias2"] = hidden2_gradient.sum(axis=0)
        hidden1_gradient = hidden2_gradient @ p["weight2"].T
        hidden1_gradient[hidden1_pre <= 0.0] = 0.0
        gradients["weight1"] = rows.T @ hidden1_gradient
        gradients["bias1"] = hidden1_gradient.sum(axis=0)
        return float(np.mean(losses)), gradients

    def copy_from(self, other):
        for name in self.PARAMETER_NAMES:
            self.parameters[name][...] = other.parameters[name]


class SegmentD3QNAgent(D3QNAgent):
    """The project's Double-DQN agent with the shared-segment network."""

    def __init__(self, segment_count, segment_features, global_features, **kwargs):
        observation_size = segment_count * segment_features + global_features
        super().__init__(observation_size, segment_count, **kwargs)
        self.online = SharedSegmentDuelingNetwork(
            segment_count, segment_features, global_features, self.hidden_size, self.rng
        )
        self.target = SharedSegmentDuelingNetwork(
            segment_count, segment_features, global_features, self.hidden_size, self.rng
        )
        self.target.copy_from(self.online)
        self._adam_mean = {k: np.zeros_like(v) for k, v in self.online.parameters.items()}
        self._adam_variance = {k: np.zeros_like(v) for k, v in self.online.parameters.items()}


class D3QNSegmentPolicy(PairSegmentMarket):
    """Choose each pair's segment with a shared-segment D3QN.

    Observation = h segment blocks + one global block.  A transition links a
    pair's observation in this round to the same searcher-good slot in the
    next round; an episode ends when one optimizer call ends.
    """

    NEEDS_ENERGY_FEATURES = True
    CHECKPOINT_SCHEMA = "rl_ring_sets/1"
    # 只看「有沒有變好」時，短測中 D3QN 學會大量做 1–2 顆的小改動：回饋上升，
    # 最終 fitness 卻比隨機差。因此改用改善幅度，並以近期正改善的平均值正規化。
    REWARD_VERSION = "pair_child_gain_normalized/1"
    GAIN_SCALE_MEMORY = 0.1
    REWARD_CAP = 5.0
    SEGMENT_FEATURE_NAMES = (
        "effective_1_2", "effective_3_4", "effective_5_plus",
        "inactive_differences", "span_length", "searcher_active_ratio",
        "contains_searcher_bottleneck", "active_energy_ratio",
    )
    GLOBAL_FEATURE_NAMES = (
        "progress", "searcher_rank", "good_rank", "good_is_better",
        "searcher_stagnation", "mean_energy_ratio", "low_energy_ratio",
    )
    LOW_ENERGY_RATIO = 0.3

    def __init__(
        self, problem, *args, training=False, model_path=None, hidden_size=64,
        replay_capacity=200000, replay_group_capacity=None, replay_warmup=1024,
        batch_size=64, gamma=0.5, learning_rate=3e-4, epsilon_start=1.0,
        epsilon_end=0.05, epsilon_decay=1.0, target_update_interval=500,
        gradient_steps=4, **kwargs,
    ):
        super().__init__(problem, *args, **kwargs)
        self.training = bool(training)
        self.model_path = None if model_path is None else Path(model_path)
        self.gradient_steps = max(1, int(gradient_steps))
        self.agent = SegmentD3QNAgent(
            self.h, len(self.SEGMENT_FEATURE_NAMES), len(self.GLOBAL_FEATURE_NAMES),
            hidden_size=hidden_size, replay_capacity=replay_capacity,
            replay_group_capacity=replay_group_capacity, replay_warmup=replay_warmup,
            batch_size=batch_size, gamma=gamma, learning_rate=learning_rate,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay, target_update_interval=target_update_interval,
            training=self.training, seed=self.seed ^ 0x5E6D3A10,
        )
        self.agent.checkpoint_metadata = self._checkpoint_metadata(problem)
        if self.model_path is not None and self.model_path.is_file():
            self.agent.load(self.model_path)
            self._check_checkpoint(problem)
        elif self.model_path is not None and not self.training:
            raise FileNotFoundError(f"{type(self).__name__} model does not exist: {self.model_path}")
        self._pending = None
        self.reset_policy_statistics()

    # ---- checkpoint -------------------------------------------------------
    def _environment(self, problem):
        contract = environment_contract(problem)
        encoding = self.ENCODING_TYPE
        contract["encoding"] = {
            "class": f"{encoding.__module__}.{encoding.__name__}",
            "source_sha256": hashlib.sha256(Path(inspect.getfile(encoding)).read_bytes()).hexdigest(),
        }
        return contract

    def _checkpoint_metadata(self, problem):
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "algorithm": type(self).__name__,
            "reward_version": self.REWARD_VERSION,
            "segment_feature_names": list(self.SEGMENT_FEATURE_NAMES),
            "global_feature_names": list(self.GLOBAL_FEATURE_NAMES),
            "segments": self.h,
            "environment": self._environment(problem),
        }

    def _check_checkpoint(self, problem):
        saved = self.agent.checkpoint_metadata
        expected = self._checkpoint_metadata(problem)
        # 編碼（SensorEncoding／C4）已在 environment 裡比對，algorithm 名稱只作紀錄。
        for key in ("schema", "reward_version", "segment_feature_names",
                    "global_feature_names", "segments"):
            if saved.get(key) != expected[key]:
                raise ValueError(f"{type(self).__name__} checkpoint {key} differs from this build")
        if saved.get("environment") != expected["environment"]:
            raise ValueError(
                "RL checkpoint environment differs from this run (fitness, routing, "
                "decode, encoding or sensing mode). Use the training configuration or retrain."
            )

    def save_model(self, path=None):
        target = Path(path) if path is not None else self.model_path
        if target is None:
            raise ValueError("no model path to save to")
        return self.agent.save(target)

    # ---- 觀察值 -----------------------------------------------------------
    def _observations(self, context, pairs):
        energy = context["energy_ratio"]
        global_shared = [float(np.mean(energy)), float(np.mean(energy < self.LOW_ENERGY_RATIO))]
        searcher_rank = _normalized_rank(self.searcher_fitness)
        good_rank = _normalized_rank(self.goods_fitness)
        rows = []
        for s_id, g_id, _, stats in pairs:
            blocks = []
            for stat in stats:
                e = stat["effective"]
                blocks += [
                    1.0 if 1 <= e <= 2 else 0.0, 1.0 if 3 <= e <= 4 else 0.0, 1.0 if e >= 5 else 0.0,
                    min(stat["inactive"], 5) / 5.0, stat["length"], stat["active_ratio"],
                    stat["bottleneck"], stat["energy"],
                ]
            blocks += [
                context["progress"], searcher_rank[s_id], good_rank[g_id],
                float(self.goods_fitness[g_id] > self.searcher_fitness[s_id]),
                min(int(self.searcher_stagnation[s_id]), self.STAGNATION_SCALE) / self.STAGNATION_SCALE,
                *global_shared,
            ]
            rows.append(blocks)
        return np.asarray(rows, dtype=np.float32)

    # ---- 回饋 -------------------------------------------------------------
    def _pair_rewards(self, pairs, child1_fitness, child2_fitness,
                      searcher_before, goods_before):
        """Normalized improvement size: 0.5 × (gain of child1 + gain of child2) / scale.

        ``scale`` is a running median-based average of this optimizer call's positive pair
        gains, so a typical improvement is worth about 1 in every phase while
        larger improvements are worth proportionally more (capped).
        """
        gains = np.array([
            0.5 * max(0.0, float(child1_fitness[s_id, g_id] - searcher_before[s_id]))
            + 0.5 * max(0.0, float(child2_fitness[s_id, g_id] - goods_before[g_id]))
            for s_id, g_id, _, _ in pairs
        ])
        positive = gains[gains > IMPROVEMENT_EPSILON]
        if len(positive):
            # 中位數：第 0 段初期不可行→可行的大跳躍不會把尺度撐大。
            observed = float(np.median(positive))
            self.gain_scale = (
                observed if self.gain_scale is None
                else (1 - self.GAIN_SCALE_MEMORY) * self.gain_scale + self.GAIN_SCALE_MEMORY * observed
            )
        if self.gain_scale is None:
            return np.zeros(len(gains))
        return np.minimum(gains / self.gain_scale, self.REWARD_CAP)

    # ---- 選擇與學習 -------------------------------------------------------
    def initialize_market(self, problem, initial_state=None):
        self._pending = None
        self.gain_scale = None
        super().initialize_market(problem, initial_state)

    def _choose_segments(self, context, pairs):
        observations = self._observations(context, pairs)
        if self._pending is not None:
            self._store(*self._pending, observations, done=False)
        self._current_observations = observations
        choices = self.agent.select_actions(observations)
        return [int(c) for c in choices]

    def _observe_round(self, context, pairs, choices, rewards, done):
        self._record_statistics(context, choices, rewards)
        pending = (self._current_observations, np.asarray(choices), rewards)
        if done:
            self._store(*pending, pending[0], done=True)
            self._pending = None
        else:
            self._pending = pending
        if self.training:
            for _ in range(self.gradient_steps):
                loss = self.agent.learn()
                if loss is not None:
                    self.loss_history.append(loss)

    def _store(self, observations, actions, rewards, next_observations, done):
        if not self.training:
            return
        mask = np.ones(self.h, dtype=bool)
        for obs, action, reward, next_obs in zip(observations, actions, rewards, next_observations):
            self.agent.remember(obs, int(action), float(reward), next_obs, bool(done), mask)

    # ---- 統計 -------------------------------------------------------------
    def reset_policy_statistics(self):
        self.segment_uses = np.zeros(self.h, dtype=int)
        self.segment_reward = np.zeros(self.h, dtype=float)
        self.effective_bucket_uses = np.zeros(4, dtype=int)
        self.effective_bucket_reward = np.zeros(4, dtype=float)
        self.progress_reward = np.zeros((3, 2), dtype=float)
        self.loss_history = []

    def _record_statistics(self, context, choices, rewards):
        phase = 0 if context["progress"] < 0.1 else (1 if context["progress"] < 0.5 else 2)
        for (_, _, _, stats), choice, reward in zip(self._last_pairs, choices, rewards):
            e = stats[choice]["effective"]
            bucket = 0 if e == 0 else (1 if e <= 2 else (2 if e <= 4 else 3))
            self.segment_uses[choice] += 1
            self.segment_reward[choice] += reward
            self.effective_bucket_uses[bucket] += 1
            self.effective_bucket_reward[bucket] += reward
            self.progress_reward[phase] += (reward, 1)

    def policy_statistics(self):
        uses = np.maximum(self.effective_bucket_uses, 1)
        return {
            "segment_uses": self.segment_uses.tolist(),
            "effective_bucket_share": (self.effective_bucket_uses / max(1, self.effective_bucket_uses.sum())).round(3).tolist(),
            "effective_bucket_reward": (self.effective_bucket_reward / uses).round(4).tolist(),
            "reward_by_phase": (self.progress_reward[:, 0] / np.maximum(self.progress_reward[:, 1], 1)).round(4).tolist(),
            "epsilon": float(self.agent.epsilon),
            "replay_size": int(len(self.agent.replay)),
            "training_steps": int(self.agent.training_steps),
            "mean_loss": float(np.mean(self.loss_history)) if self.loss_history else None,
        }



# =============================================================================
# v2（2026-10-04）：目標＝段末最好解的 fitness。
# * 輸出不變：每組交易選 1 個區段（範圍仍用 80/20 規則預先抽好）。
# * 輸入：每個區段 9 個特徵＋7 個共同特徵，整批向量化計算。新增「最慘 target 的
#   成本占比」與「全網成本占比」，直接對應 fitness v2 的 0.7／0.3 兩項；拿掉 lifetime
#   取向的「是否包含瓶頸 sensor」。
# * 回饋：分級（參考 DE-DDQN 的 R2）。每個子代超越回合前的最好解 = 10、比自己的
#   父代好 = 1、其他 = 0；交易回饋 = 0.5 × child1 + 0.5 × child2。不需要正規化。
# * D3QN 預設 γ = 0（選區段的效果是即時的，當作 contextual bandit）。
# =============================================================================
class PairSegmentMarketV2(PairSegmentMarket):
    """v2 features (vectorized) and the tiered reward."""

    SEGMENT_FEATURE_NAMES = (
        "effective_1_2", "effective_3_4", "effective_5_plus",
        "inactive_differences", "span_length", "searcher_active_ratio",
        "worst_target_cost_share", "network_cost_share", "active_energy_ratio",
    )
    GLOBAL_FEATURE_NAMES = (
        "progress", "searcher_rank", "good_rank", "good_is_better",
        "searcher_stagnation", "mean_energy_ratio", "low_energy_ratio",
    )
    LOW_ENERGY_RATIO = 0.3
    TIER_BEST = 10.0
    TIER_PARENT = 1.0

    def _round_context(self, problem):
        energy = np.maximum(np.asarray(problem.energy, dtype=float), 0.0)
        energy_ratio = energy / max(float(problem.initial_energy), 1e-15)
        searcher_info = [self._decoded(problem, s, need_cost=True) for s in self.searchers]
        active_s = np.asarray([info["active"] for info in searcher_info])
        cost_s = np.maximum(np.asarray([info["cost"] for info in searcher_info]), 0.0)
        active_g = np.asarray([self._decoded(problem, g)["active"] for g in self.goods])

        # fitness v2 的最慘 target：鄰近 sensor（target_sensor_mask）總成本 ÷ 總電量最大者。
        mask = np.asarray(problem.target_sensor_mask, dtype=float)
        target_energy = mask @ energy
        target_cost = cost_s @ mask.T
        depletion = np.ones_like(target_cost)
        np.divide(target_cost, target_energy[None, :], out=depletion,
                  where=target_energy[None, :] > 1e-15)
        worst_mask = mask[np.argmax(depletion, axis=1)] > 0
        return {
            "progress": min(1.0, self.evatime / max(1, self.evaluation_limit)),
            "energy_ratio": energy_ratio,
            "active_s": active_s, "active_g": active_g,
            "cost_s": cost_s, "worst_cost_s": cost_s * worst_mask,
        }

    def _build_pairs(self, context):
        n, w, h = self.n, self.w, self.h
        # 抽範圍的順序與 v1 相同：投資者 → 商品 → 區段。
        spans = [[self.select_sensor_span(region) for region in range(h)]
                 for _ in range(n * w)]
        left = np.array([[span[1] for span in row] for row in spans])
        right = np.array([[span[2] for span in row] for row in spans])
        s_ids = np.repeat(np.arange(n), w)
        g_ids = np.tile(np.arange(w), n)

        codes_s = np.asarray([np.asarray(s.code) for s in self.searchers])
        codes_g = np.asarray([np.asarray(g.code) for g in self.goods])
        differ = ((codes_s[s_ids, 0::2] != codes_g[g_ids, 0::2])
                  | (codes_s[s_ids, 1::2] != codes_g[g_ids, 1::2]))
        either_on = context["active_s"][s_ids] | context["active_g"][g_ids]

        def prefix(values):
            values = np.asarray(values, dtype=float)
            return np.concatenate([np.zeros(values.shape[:-1] + (1,)), np.cumsum(values, axis=-1)], axis=-1)

        def span_sum(cumulative, rows):
            return (np.take_along_axis(cumulative[rows], right, axis=1)
                    - np.take_along_axis(cumulative[rows], left, axis=1))

        pair_rows = np.arange(n * w)
        effective = span_sum(prefix(differ & either_on), pair_rows)
        inactive = span_sum(prefix(differ & ~either_on), pair_rows)
        active_count = span_sum(prefix(context["active_s"]), s_ids)
        length = (right - left).astype(float)
        cost_total = context["cost_s"].sum(axis=1)[s_ids][:, None]
        worst_total = context["worst_cost_s"].sum(axis=1)[s_ids][:, None]
        cost_in = span_sum(prefix(context["cost_s"]), s_ids)
        worst_in = span_sum(prefix(context["worst_cost_s"]), s_ids)
        energy_in = span_sum(prefix(context["active_s"] * context["energy_ratio"][None, :]), s_ids)

        segment = np.zeros((n * w, h, len(self.SEGMENT_FEATURE_NAMES)), dtype=np.float32)
        segment[..., 0] = (effective >= 1) & (effective <= 2)
        segment[..., 1] = (effective >= 3) & (effective <= 4)
        segment[..., 2] = effective >= 5
        segment[..., 3] = np.minimum(inactive, 5) / 5.0
        segment[..., 4] = length / self.max_region_length
        segment[..., 5] = active_count / np.maximum(length, 1)
        segment[..., 6] = np.divide(worst_in, worst_total, out=np.zeros_like(worst_in), where=worst_total > 1e-15)
        segment[..., 7] = np.divide(cost_in, cost_total, out=np.zeros_like(cost_in), where=cost_total > 1e-15)
        segment[..., 8] = np.divide(energy_in, active_count, out=np.zeros_like(energy_in), where=active_count > 0)

        energy_ratio = context["energy_ratio"]
        searcher_rank = _normalized_rank(self.searcher_fitness)
        good_rank = _normalized_rank(self.goods_fitness)
        shared = np.zeros((n * w, len(self.GLOBAL_FEATURE_NAMES)), dtype=np.float32)
        shared[:, 0] = context["progress"]
        shared[:, 1] = searcher_rank[s_ids]
        shared[:, 2] = good_rank[g_ids]
        shared[:, 3] = self.goods_fitness[g_ids] > self.searcher_fitness[s_ids]
        shared[:, 4] = np.minimum(self.searcher_stagnation[s_ids], self.STAGNATION_SCALE) / self.STAGNATION_SCALE
        shared[:, 5] = float(np.mean(energy_ratio))
        shared[:, 6] = float(np.mean(energy_ratio < self.LOW_ENERGY_RATIO))

        self._segment_features, self._shared_features = segment, shared
        self._effective_counts = effective
        return [(int(s), int(g), spans[k], None) for k, (s, g) in enumerate(zip(s_ids, g_ids))]

    def _pair_rewards(self, pairs, child1_fitness, child2_fitness,
                      searcher_before, goods_before):
        """Per child: beats the pre-round best = 10, beats its parent = 1, else 0."""
        best = self._best_before + IMPROVEMENT_EPSILON

        def tier(child, parent):
            return np.where(child > best, self.TIER_BEST,
                            np.where(child > parent + IMPROVEMENT_EPSILON, self.TIER_PARENT, 0.0))

        s_ids = np.array([p[0] for p in pairs])
        g_ids = np.array([p[1] for p in pairs])
        c1 = child1_fitness[s_ids, g_ids]
        c2 = child2_fitness[s_ids, g_ids]
        return 0.5 * tier(c1, searcher_before[s_ids]) + 0.5 * tier(c2, goods_before[g_ids])


class LinUCBSegmentPolicyV2(PairSegmentMarketV2):
    """Shared LinUCB on the v2 features: constant + 9 segment + 7 shared + progress × small."""

    ALPHA = 0.3
    RIDGE = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size = 1 + len(self.SEGMENT_FEATURE_NAMES) + len(self.GLOBAL_FEATURE_NAMES) + 1
        self.linucb_A = self.RIDGE * np.eye(size)
        self.linucb_b = np.zeros(size)

    def _choose_segments(self, context, pairs):
        segment, shared = self._segment_features, self._shared_features
        count, h = segment.shape[:2]
        X = np.concatenate([
            np.ones((count, h, 1)),
            segment,
            np.repeat(shared[:, None, :], h, axis=1),
            (shared[:, 0][:, None] * segment[..., 0])[..., None],
        ], axis=2).astype(float)
        A_inv = np.linalg.inv(self.linucb_A)
        theta = A_inv @ self.linucb_b
        scores = X @ theta + self.ALPHA * np.sqrt(np.einsum("phi,ij,phj->ph", X, A_inv, X))
        choices = []
        for row in scores:
            best = np.flatnonzero(row >= row.max() - 1e-12)
            choices.append(int(best[self.random.randrange(len(best))]))
        self._chosen_features = X[np.arange(count), choices]
        return choices

    def _observe_round(self, context, pairs, choices, rewards, done):
        X = self._chosen_features
        self.linucb_A += X.T @ X
        self.linucb_b += X.T @ np.asarray(rewards, dtype=float)


class D3QNSegmentPolicyV2(PairSegmentMarketV2, D3QNSegmentPolicy):
    """Shared-segment D3QN on the v2 features with the tiered reward (γ = 0 by default)."""

    CHECKPOINT_SCHEMA = "rl_ring_sets/2"
    REWARD_VERSION = "pair_child_tiered/1"

    def __init__(self, problem, *args, gamma=0.0, **kwargs):
        super().__init__(problem, *args, gamma=gamma, **kwargs)

    def _observations(self, context, pairs):
        segment, shared = self._segment_features, self._shared_features
        return np.concatenate([segment.reshape(len(segment), -1), shared], axis=1)

    def _observe_round(self, context, pairs, choices, rewards, done):
        if self.agent.gamma == 0.0:
            # γ = 0：每回合的決策各自獨立，直接以 done 存入，不需要下一個狀態。
            self._record_statistics(context, choices, rewards)
            observations = self._current_observations
            self._store(observations, np.asarray(choices), rewards, observations, done=True)
            self._pending = None
            if self.training:
                for _ in range(self.gradient_steps):
                    loss = self.agent.learn()
                    if loss is not None:
                        self.loss_history.append(loss)
            return
        super()._observe_round(context, pairs, choices, rewards, done)

    def _record_statistics(self, context, choices, rewards):
        phase = 0 if context["progress"] < 0.1 else (1 if context["progress"] < 0.5 else 2)
        chosen = self._effective_counts[np.arange(len(choices)), choices]
        buckets = np.select([chosen == 0, chosen <= 2, chosen <= 4], [0, 1, 2], default=3)
        for choice, bucket, reward in zip(choices, buckets, rewards):
            self.segment_uses[choice] += 1
            self.segment_reward[choice] += reward
            self.effective_bucket_uses[bucket] += 1
            self.effective_bucket_reward[bucket] += reward
            self.progress_reward[phase] += (reward, 1)
        self.reward_tier_counts = getattr(self, "reward_tier_counts", np.zeros(6, dtype=int))
        for value in rewards:
            self.reward_tier_counts[int(np.searchsorted([0.0, 0.5, 1.0, 5.0, 5.5, 10.0], value))] += 1

    def reset_policy_statistics(self):
        super().reset_policy_statistics()
        self.reward_tier_counts = np.zeros(6, dtype=int)

    def policy_statistics(self):
        stats = super().policy_statistics()
        total = max(1, int(self.reward_tier_counts.sum()))
        stats["reward_share"] = dict(zip(("0", "0.5", "1", "5", "5.5", "10"),
                                         (self.reward_tier_counts / total).round(4).tolist()))
        return stats


def _normalized_rank(values):
    """0 = worst, 1 = best; ties share the lower rank."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.ones(len(values))
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order / (len(values) - 1)


__all__ = [
    "D3QNSegmentPolicy",
    "D3QNSegmentPolicyV2",
    "LinUCBSegmentPolicy",
    "LinUCBSegmentPolicyV2",
    "PairSegmentMarketV2",
    "PairSegmentMarket",
    "SegmentD3QNAgent",
    "SharedSegmentDuelingNetwork",
]
