"""LinUCB-Ring-SE：Ring_SE（SensorEncoding），每組 searcher-good 用 LinUCB 選區段。

區段內範圍仍用 Ring_SE 原本的 80/20 規則；逐組選區段的流程見
:mod:`ring_segment_policy`。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.ring_se.ring_segment_policy import PairSegmentMarket


class LinUCB_Ring_SE(PairSegmentMarket):
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

    def __init__(self, problem, n=8, h=4, w=8, mu=0.4, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
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


__all__ = ["LinUCB_Ring_SE"]
