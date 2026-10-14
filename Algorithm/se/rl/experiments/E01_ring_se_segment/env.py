"""E01 環境：Ring_SE 的每組 searcher-good 交易由外部策略選段。

* 狀態：每組交易 60 維 = 4 段 × 14 ‖ 全域 4（定義見 d3qn/README.md 第 1 節），段的位置固定。
* 動作：4 段選 1；每段的候選範圍照 Ring_SE 的 80/20 規則各抽一個（PairSegmentMarket）。
* 策略：``policy`` 可以是 "random"（等同 ring_se）、"oracle"（作弊版：偷看 4 段只交換的結果）、
  或一個 callable(states (P, 60)) -> actions (P,)。
* 獎勵：Δ = max(0, f(c1) − f(searcher)) + max(0, f(c2) − f(good))；Δ ≤ 1e-12 → 0，
  否則 0.5 + 0.5 × （Δ 在本 episode 最近 256 個正 Δ 中的百分位，視窗回合開始時凍結）。
* 轉移：同一個 searcher 編號、同一個 good 位置的下一回合；該段最佳化的最後一回合為終止。
  每個 episode（一段最佳化）結束時交給 ``recorder.add_episode``，裡面同時算 1-step 與 n-step。

交換、突變、EN 更新都和 Ring_SE 相同；評估照常計入預算（作弊版的偷看不計）。
"""
from __future__ import annotations

import collections

import numpy as np

from Algorithm.se.market_components import accept_good_children, accept_searcher_children
from Algorithm.se.ring_se.Ring_SE import Ring_SE
from Algorithm.se.ring_se.ring_segment_policy import IMPROVEMENT_EPSILON, PairSegmentMarket
from State.Encoding import swap_segment

H = 4
SEG_DIM = 14
GLOBAL_DIM = 4
STATE_DIM = H * SEG_DIM + GLOBAL_DIM  # 60
REWARD_WINDOW = 256
SUCCESS_WINDOW = 50
BOTTLENECK_COUNT = 5


class EpisodeRecorder:
    """收集一段最佳化的 (s, a, r)，結束時算出 1-step 與 n-step 轉移。"""

    def __init__(self, gamma=0.5, n_step=3):
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.transitions = []  # list of dict arrays（每個 episode 一份）

    def add_episode(self, trajectories):
        """trajectories: {key: [(state, action, reward), ...]}（依回合順序）。"""
        rows = {k: [] for k in ("s", "a", "r", "s1", "d1", "rn", "sn", "dn", "gn")}
        zero = np.zeros(STATE_DIM, dtype=np.float32)
        for steps in trajectories.values():
            length = len(steps)
            for t, (state, action, reward) in enumerate(steps):
                rows["s"].append(state)
                rows["a"].append(action)
                rows["r"].append(reward)
                rows["s1"].append(steps[t + 1][0] if t + 1 < length else zero)
                rows["d1"].append(float(t + 1 >= length))
                m = min(self.n_step, length - t)
                rows["rn"].append(sum(self.gamma ** j * steps[t + j][2] for j in range(m)))
                rows["sn"].append(steps[t + m][0] if t + m < length else zero)
                rows["dn"].append(float(t + m >= length))
                rows["gn"].append(self.gamma ** m)
        if rows["s"]:
            self.transitions.append({
                "s": np.asarray(rows["s"], dtype=np.float32),
                "a": np.asarray(rows["a"], dtype=np.int64),
                "r": np.asarray(rows["r"], dtype=np.float32),
                "s1": np.asarray(rows["s1"], dtype=np.float32),
                "d1": np.asarray(rows["d1"], dtype=np.float32),
                "rn": np.asarray(rows["rn"], dtype=np.float32),
                "sn": np.asarray(rows["sn"], dtype=np.float32),
                "dn": np.asarray(rows["dn"], dtype=np.float32),
                "gn": np.asarray(rows["gn"], dtype=np.float32),
            })

    def pop_all(self):
        out, self.transitions = self.transitions, []
        return out


class Ring_SE_E01(PairSegmentMarket, Ring_SE):
    """Ring_SE whose per-pair segment is chosen by an external policy (random / oracle / learned)."""

    NEEDS_ENERGY_FEATURES = False  # 能量、耗能特徵由本類別自己算

    def __init__(self, problem, policy="random", recorder=None, **kwargs):
        super().__init__(problem, **kwargs)
        self.policy = policy
        self.recorder = recorder
        self.round_stats = []

    # ---- episode 狀態 -------------------------------------------------------
    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self._info_cache = {}
        self._reward_window = collections.deque(maxlen=REWARD_WINDOW)
        self._success = [collections.deque(maxlen=SUCCESS_WINDOW) for _ in range(self.h)]
        self._trajectories = collections.defaultdict(list)
        self.round_stats = []

    def _searcher_info(self, problem, candidate):
        """解碼一次：開啟、等級、每顆耗能、電量÷耗能最低的 5 顆、只有自己覆蓋的 target 數。"""
        key = np.asarray(candidate.code).tobytes()
        info = self._info_cache.get(key)
        if info is not None:
            return info
        n = problem.SENSOR_NUMBER
        state = candidate.decode(problem)
        levels = np.asarray(state.levels[:n], dtype=int)
        active = levels > 0
        radii = np.asarray(problem.state_radius(state), dtype=float)
        cost = np.asarray(problem.calculate_total_cost(state, sensing_radii=radii), dtype=float)[:n]
        cost = np.where(np.isfinite(cost), cost, 0.0)
        energy = np.maximum(np.asarray(problem.energy[:n], dtype=float), 0.0)
        bottleneck = np.zeros(n, dtype=bool)
        spending = np.flatnonzero(active & (cost > 1e-15))
        if len(spending):
            order = spending[np.argsort(energy[spending] / cost[spending], kind="stable")]
            bottleneck[order[:BOTTLENECK_COUNT]] = True
        coverage = np.asarray(problem.coverage_table) > 0  # (T, S, L)
        on = np.flatnonzero(active)
        covered = coverage[:, on, levels[on]]  # (T, |on|)
        counts = covered.sum(axis=1)
        single = np.zeros(n, dtype=float)
        single[on] = (covered & (counts[:, None] == 1)).sum(axis=0)
        info = {"active": active, "cost": cost, "bottleneck": bottleneck, "single": single,
                "total_cost": max(float(cost.sum()), 1e-15)}
        self._info_cache[key] = info
        return info

    # ---- 狀態 ---------------------------------------------------------------
    def _states(self, problem, context, pairs):
        n = problem.SENSOR_NUMBER
        energy_ratio = np.maximum(np.asarray(problem.energy[:n], dtype=float), 0.0) / max(
            float(problem.initial_energy), 1e-15)
        alive = np.asarray(problem.energy[:n], dtype=float) >= problem.liveJ
        mean_alive_energy = float(energy_ratio[alive].mean()) if np.any(alive) else 0.0
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        success = [float(np.mean(d)) if len(d) else 0.5 for d in self._success]
        gaps = np.array([self.goods_fitness[g] - self.searcher_fitness[s] for s, g, _, _ in pairs])
        gap_rank = np.argsort(np.argsort(gaps, kind="stable"), kind="stable") / max(1, len(pairs) - 1)

        infos = [self._searcher_info(problem, s) for s in self.searchers]
        states = np.zeros((len(pairs), STATE_DIM), dtype=np.float32)
        for i, (s_id, g_id, spans, stats) in enumerate(pairs):
            info = infos[s_id]
            g_active = context["goods_active"][g_id]
            for k, ((_, left, right), stat) in enumerate(zip(spans, stats)):
                eff = stat["effective"]
                on = info["active"][left:right]
                e_on = energy_ratio[left:right][on]
                n_on = int(on.sum())
                row = states[i, k * SEG_DIM:(k + 1) * SEG_DIM]
                row[0 if eff == 0 else 1 if eff <= 2 else 2 if eff <= 4 else 3] = 1.0
                row[4] = min(stat["inactive"], 5) / 5.0
                row[5] = stat["length"]
                row[6] = stat["active_ratio"]
                row[7] = float(g_active[left:right].mean()) if right > left else 0.0
                row[8] = float(e_on.min()) if n_on else 1.0
                row[9] = float(e_on.mean()) if n_on else 1.0
                row[10] = info["bottleneck"][left:right].sum() / BOTTLENECK_COUNT
                row[11] = info["cost"][left:right].sum() / info["total_cost"]
                row[12] = min(1.0, info["single"][left:right].sum() / max(1, n_on))
                row[13] = success[k]
            base = H * SEG_DIM
            states[i, base + 0] = progress
            states[i, base + 1] = min(int(self.searcher_stagnation[s_id]), 20) / 20.0
            states[i, base + 2] = gap_rank[i]
            states[i, base + 3] = mean_alive_energy
        return states

    # ---- 選段 ---------------------------------------------------------------
    def _peek_choices(self, pairs, searcher_before, goods_before):
        """作弊版：4 段只交換的子代偷看評估（不計預算、不更新最佳），選 Δ 最大、同分比 f(child2)。"""
        rows, crossed = [], []
        for s_id, g_id, spans, _ in pairs:
            row, kids = [], []
            for span in spans:
                c1, c2 = swap_segment(self.searchers[s_id], self.goods[g_id], span[1] * 2, span[2] * 2)
                row += [c1, c2]
                kids.append((c1, c2))
            rows.append(row)
            crossed.append(kids)
        fit = np.array([[float(np.sum(o)) for _, _, o in res] for res in self._route_to_pool(rows)],
                       dtype=float).reshape(len(pairs), self.h, 2)
        choices = []
        for i, (s_id, g_id, _, _) in enumerate(pairs):
            f1, f2 = fit[i, :, 0], fit[i, :, 1]
            delta = np.maximum(0.0, f1 - searcher_before[s_id]) + np.maximum(0.0, f2 - goods_before[g_id])
            delta = np.where(delta > IMPROVEMENT_EPSILON, delta, 0.0)
            choices.append(int(np.lexsort((f2, delta))[-1]))
        return np.asarray(choices, dtype=int), crossed

    # ---- 一回合 -------------------------------------------------------------
    def vision_search(self, problem):
        goods_before = self.goods_fitness.copy()
        searcher_before = self.searcher_fitness.copy()
        context = self._round_context(problem)
        pairs = self._build_pairs(context)
        states = self._states(problem, context, pairs)

        crossed = None
        if self.policy == "random":
            actions = np.array([self.random.randrange(self.h) for _ in pairs], dtype=int)
        elif self.policy == "oracle":
            actions, crossed = self._peek_choices(pairs, searcher_before, goods_before)
        else:
            actions = np.asarray(self.policy(states), dtype=int)

        children = [[] for _ in range(self.n)]
        for i, ((s_id, g_id, spans, _), k) in enumerate(zip(pairs, actions)):
            span = spans[int(k)]
            if crossed is not None:
                kids = [c.copy() for c in crossed[i][int(k)]]
            else:
                kids = list(swap_segment(self.searchers[s_id], self.goods[g_id], span[1] * 2, span[2] * 2))
            for child in kids:
                if self.random.random() < self.mutation_rate:
                    self.mutate_candidate(problem, child, sensor_span=span)
            children[s_id] += kids

        child_fitness = self.evaluate_investments(problem, children)
        f1 = child_fitness[:, 0::2]
        f2 = child_fitness[:, 1::2]

        # 每步獎勵（視窗在回合開始時凍結）
        window = np.sort(np.asarray(self._reward_window, dtype=float))
        deltas = np.array([max(0.0, f1[s, g] - searcher_before[s]) + max(0.0, f2[s, g] - goods_before[g])
                           for s, g, _, _ in pairs])
        rewards = np.zeros(len(pairs), dtype=np.float32)
        for i, d in enumerate(deltas):
            if d > IMPROVEMENT_EPSILON:
                pct = np.searchsorted(window, d, side="right") / len(window) if len(window) else 0.5
                rewards[i] = 0.5 + 0.5 * pct
        for d in deltas:
            if d > IMPROVEMENT_EPSILON:
                self._reward_window.append(float(d))
        for k, d in zip(actions, deltas):
            self._success[int(k)].append(float(d > IMPROVEMENT_EPSILON))

        accept_searcher_children(self, children, f1, searcher_before)
        accept_good_children(self.goods, self.goods_fitness, goods_before, children,
                             f2, range(self.n), self.GOOD_ACCEPT_TOLERANCE)
        improved = self.searcher_fitness > searcher_before + IMPROVEMENT_EPSILON
        self.searcher_stagnation = np.where(improved, 0, self.searcher_stagnation + 1)

        for (s_id, g_id, _, _), state, action, reward in zip(pairs, states, actions, rewards):
            self._trajectories[(s_id, g_id)].append((state, int(action), float(reward)))
        self.round_stats.append({"mean_reward": float(rewards.mean()),
                                 "segment_counts": np.bincount(actions, minlength=self.h).tolist()})

    def search(self, problem, budget, state=None):
        result = super().search(problem, budget, state)
        if self.recorder is not None and self._trajectories:
            self.recorder.add_episode(self._trajectories)
        self._trajectories = collections.defaultdict(list)
        return result


__all__ = ["EpisodeRecorder", "Ring_SE_E01", "STATE_DIM", "H"]
