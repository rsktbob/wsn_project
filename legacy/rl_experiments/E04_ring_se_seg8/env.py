"""E04 環境：ring_se 環固定切 8 段，每組 searcher-good 交易由外部策略選 1 段，整段交換、段內突變。

sensor 順序是「由內到外的距離環、環內依角度」（Problem.sensor_ring_order），所以段 0 最內圈、段 7 最外圈，
段 k 和 k±1 相鄰（段 0 和段 7 不相鄰）。

狀態（每組交易一個，攤平成 STATE_DIM 維；網路自己拆回去）：
    tokens (8, 13, 25)：每段每顆 sensor 一個 token，同 E03（S 解碼 10 ‖ G 解碼 10 ‖ E/E0、離 BS 距離 ‖ 差異旗標 3）
    mask   (8, 13)    ：真的 sensor 為 1，補的空位為 0（12 顆的段最後一格是空位）
    seg    (8, 7)     ：瓶頸分數 b_k、有效差異數、無效差異數、範圍外影響 4（同 E03）
    glob   (10,)      ：同 E03
策略 policy："random"（等同 8 段整段隨機）、"bottleneck"（選 b_k 最大、同分隨機）、
    "oracle"（作弊版：偷看 8 段只交換的結果，不計預算、不更新最佳，選 Δ 最大、同分比 f(child2)，再照常突變）、
    或 callable(states (P, STATE_DIM)) -> actions (P,)。
recorder 不是 None 時（作弊版或隨機），每回合存下狀態與 8 段只交換的 Δ、f(child2)、作弊版會選的段、實際選的段，
供監督式學習。隨機＋recorder＝「作弊版只標答案、族群照 8 段隨機推進」（偷看不用亂數，推進和純隨機完全相同）。
獎勵：Δ = max(0, f(c1) − f(searcher)) + max(0, f(c2) − f(good))；Δ ≤ 1e-12 → 0，
    否則 min(Δ ÷ 本回合正 Δ 的中位數, 5)。
轉移：同一個 searcher 編號、同一個 good 位置的下一回合（同 E01）；這段最佳化的最後一回合為終止。
交換、突變、EN 更新都和 Ring_SE 相同；評估照常計入預算。
"""
from __future__ import annotations

import collections
import importlib.util
from pathlib import Path

import numpy as np

from Algorithm.se.ring_se.ring_segment_policy import IMPROVEMENT_EPSILON
from State.Encoding import swap_segment

_E03 = Path(__file__).resolve().parent.parent / "E03_ring_se_decoded"
_spec = importlib.util.spec_from_file_location("e03_env", _E03 / "env.py")
_e03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e03)
build_tokens, effective_mask, TIGHT = _e03.build_tokens, _e03.effective_mask, _e03.TIGHT

H = 8
LMAX = 13
TOKEN_DIM = _e03.TOKEN_DIM          # 25
SEG_DIM = 7
GLOBAL_DIM = _e03.GLOBAL_DIM        # 10
TOKENS_SIZE = H * LMAX * TOKEN_DIM  # 2600
MASK_SIZE = H * LMAX                # 104
SEG_SIZE = H * SEG_DIM              # 56
STATE_DIM = TOKENS_SIZE + MASK_SIZE + SEG_SIZE + GLOBAL_DIM  # 2770
REWARD_CAP = 5.0


def split_state(states):
    """(P, STATE_DIM) → tokens (P, 8, 13, 25)、mask (P, 8, 13)、seg (P, 8, 7)、glob (P, 10)。numpy／torch 都可。"""
    a = TOKENS_SIZE
    b = a + MASK_SIZE
    c = b + SEG_SIZE
    P = states.shape[0]
    return (states[:, :a].reshape(P, H, LMAX, TOKEN_DIM), states[:, a:b].reshape(P, H, LMAX),
            states[:, b:c].reshape(P, H, SEG_DIM), states[:, c:])


class Ring_SE_E04(_e03.Ring_SE_E03):
    """h=8、整段範圍；沿用 E03 的解碼快取（_solution）。"""

    def __init__(self, problem, policy="random", recorder=None, **kwargs):
        kwargs.setdefault("h", H)
        super().__init__(problem, policy=policy, recorder=recorder, **kwargs)
        if self.h != H:
            raise ValueError("E04 needs h=8")

    def select_sensor_span(self, region):
        start, end = self.region_sensor_bounds[self._resolve_region(region)]
        return "whole", start, end

    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self._trajectories = collections.defaultdict(list)

    # ---- 狀態 -------------------------------------------------------------
    def _states(self, problem, pairs):
        n = problem.SENSOR_NUMBER
        energy = np.maximum(np.asarray(problem.energy[:n], dtype=float), 0.0)
        e_ratio = energy / max(float(problem.initial_energy), 1e-15)
        dist = np.asarray(problem.distances)[:n, problem.BSID]
        sensor = np.stack([e_ratio, dist / max(float(dist.max()), 1e-15)], axis=1).astype(np.float32)
        alive = energy >= problem.liveJ
        mean_alive = float(e_ratio[alive].mean()) if np.any(alive) else 0.0
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        gaps = np.array([self.goods_fitness[g] - self.searcher_fitness[s] for s, g, _, _ in pairs])
        gap_rank = np.argsort(np.argsort(gaps, kind="stable"), kind="stable") / max(1, len(pairs) - 1)
        S = [self._solution(problem, x) for x in self.searchers]
        G = [self._solution(problem, x) for x in self.goods]
        bounds = self.region_sensor_bounds

        P = len(pairs)
        tokens = np.zeros((P, H, LMAX, TOKEN_DIM), dtype=np.float32)
        mask = np.zeros((P, H, LMAX), dtype=np.float32)
        seg = np.zeros((P, H, SEG_DIM), dtype=np.float32)
        glob = np.zeros((P, GLOBAL_DIM), dtype=np.float32)
        bottleneck = np.zeros((P, H), dtype=float)
        for i, (s_id, g_id, _, _) in enumerate(pairs):
            si, gi = S[s_id], G[g_id]
            differ = np.any(si["genes"] != gi["genes"], axis=1)
            either_on = (si["feat"][:, 0] > 0) | (gi["feat"][:, 0] > 0)
            for k, (left, right) in enumerate(bounds):
                L = right - left
                tokens[i, k, :L] = build_tokens(si["feat"], gi["feat"], sensor, si["genes"], gi["genes"], left, right)
                mask[i, k, :L] = 1.0
                eff = effective_mask(si["feat"], gi["feat"], si["genes"], gi["genes"], left, right)
                b = float((si["feat"][left:right, TIGHT] + gi["feat"][left:right, TIGHT])[eff].sum())
                bottleneck[i, k] = b
                inside = np.zeros(n, dtype=bool)
                inside[left:right] = True
                row = seg[i, k]
                row[0] = b
                row[1] = eff.sum()
                row[2] = (differ & ~either_on)[left:right].sum()
                for col, sol in ((0, si), (1, gi)):
                    through = sol["depends"][:, inside].any(axis=1) & ~inside & (sol["feat"][:, 0] > 0)
                    row[3 + col] = through.sum() / n
                    cov_in = sol["cov"][:, inside].any(axis=1)
                    backup = sol["cov"][:, ~inside].any(axis=1)
                    row[5 + col] = (cov_in & ~backup).sum() / max(1, int(cov_in.sum()))
            lives = (si["min_life"], gi["min_life"])
            finite = [x for x in lives if np.isfinite(x)]
            life_gap = ((lives[0] - lives[1]) / max(finite)) if len(finite) == 2 and max(finite) > 0 else 0.0
            glob[i] = (progress, min(int(self.searcher_stagnation[s_id]), 20) / 20.0, mean_alive, gap_rank[i],
                       si["coverage"], gi["coverage"], np.clip(life_gap, -1, 1), si["n_active"] / n,
                       gi["n_active"] / n, float(differ.mean()))
        states = np.concatenate([tokens.reshape(P, -1), mask.reshape(P, -1), seg.reshape(P, -1), glob], axis=1)
        return states, bottleneck

    def _peek(self, pairs):
        """8 段只交換的子代偷看評估（不計預算、不更新最佳）→ fit (P, 8, 2)、交換後的子代。"""
        rows, crossed = [], []
        for s_id, g_id, spans, _ in pairs:
            row, kids = [], []
            for span in spans:
                c1, c2 = swap_segment(self.searchers[s_id], self.goods[g_id], span[1] * 2, span[2] * 2)
                row += [c1, c2]
                kids.append((c1, c2))
            rows.append(row)
            crossed.append(kids)
        fit = np.array([[float(np.sum(o)) for _, _, o in res] for res in self.evaluator.route(rows)],
                       dtype=float).reshape(len(pairs), H, 2)
        return fit, crossed

    # ---- 一回合 -------------------------------------------------------------
    def vision_search(self, problem):
        goods_before = self.goods_fitness.copy()
        searcher_before = self.searcher_fitness.copy()
        context = self._round_context(problem)
        pairs = self._build_pairs(context)
        states, bottleneck = self._states(problem, pairs)

        crossed = None
        if self.policy == "oracle" or (self.policy == "random" and self.recorder is not None):
            fit, crossed = self._peek(pairs)
            peek = (np.maximum(0.0, fit[:, :, 0] - searcher_before[[p[0] for p in pairs]][:, None])
                    + np.maximum(0.0, fit[:, :, 1] - goods_before[[p[1] for p in pairs]][:, None]))
            peek = np.where(peek > IMPROVEMENT_EPSILON, peek, 0.0)
            oracle_choice = np.array([int(np.lexsort((fit[i, :, 1], peek[i]))[-1]) for i in range(len(pairs))])
        if self.policy == "random":
            actions = np.array([self.random.randrange(H) for _ in pairs], dtype=int)
        elif self.policy == "oracle":
            actions = oracle_choice
        elif self.policy == "bottleneck":
            actions = []
            for row in bottleneck:
                best = np.flatnonzero(row >= row.max() - 1e-12)
                actions.append(int(best[self.random.randrange(len(best))]))
            actions = np.asarray(actions, dtype=int)
        else:
            actions = np.asarray(self.policy(states), dtype=int)

        if crossed is not None and self.recorder is not None:
            self.recorder.add_round(states, peek, fit[:, :, 1], oracle_choice, actions)

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
        deltas = np.array([max(0.0, f1[s, g] - searcher_before[s]) + max(0.0, f2[s, g] - goods_before[g])
                           for s, g, _, _ in pairs])
        positive = deltas > IMPROVEMENT_EPSILON
        rewards = np.zeros(len(pairs), dtype=np.float32)
        if positive.any():
            rewards[positive] = np.minimum(deltas[positive] / np.median(deltas[positive]), REWARD_CAP)

        self.accept_searcher_children(children, f1, searcher_before)
        self.accept_good_children(self.goods, self.goods_fitness, goods_before, children,
                             f2, range(self.n))
        improved = self.searcher_fitness > searcher_before + IMPROVEMENT_EPSILON
        self.searcher_stagnation = np.where(improved, 0, self.searcher_stagnation + 1)

        for (s_id, g_id, _, _), state, action, reward in zip(pairs, states, actions, rewards):
            self._trajectories[(s_id, g_id)].append((state, int(action), float(reward)))
        self.round_stats.append({"mean_reward": float(rewards.mean()), "improve_rate": float(positive.mean()),
                                 "segment_counts": np.bincount(actions, minlength=H).tolist()})

    def pop_trajectories(self):
        """這段最佳化的 {(s_id, g_id): [(state, action, reward), ...]}（依回合順序），取出後清空。"""
        out, self._trajectories = self._trajectories, collections.defaultdict(list)
        return out


class LabelRecorder:
    """每回合：狀態 (P, STATE_DIM) float16、8 段只交換的 Δ (P, 8)、f(child2) (P, 8)、
    作弊版會選的段 choice、實際選的段 taken（作弊版推進時兩者相同）。"""

    def __init__(self):
        self.rounds = []

    def add_round(self, states, delta, f2, choice, taken):
        self.rounds.append({"states": states.astype(np.float16), "delta": delta.astype(np.float32),
                            "f2": f2, "choice": np.asarray(choice, dtype=np.int8),
                            "taken": np.asarray(taken, dtype=np.int8)})


__all__ = ["Ring_SE_E04", "LabelRecorder", "split_state", "H", "LMAX", "TOKEN_DIM", "SEG_DIM", "GLOBAL_DIM", "STATE_DIM"]
