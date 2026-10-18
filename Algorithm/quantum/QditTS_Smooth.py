"""Qdit-TS with order-aware smoothing, done with amplitudes and Givens rotations.

Same register (amplitudes, one row per SensorEncoding gene, 6 options per
sensing level, 10 per priority rank, uniform prior 1/sqrt(d)), observation and
loop as QditTS.  The update spreads the rotation over the neighbourhoods of
the best option b and the worst option w:
    g = kernel centred on b, l = kernel centred on w (each renormalized to 1),
    for every pair (i, j), i != j, with g_i > 0 and l_j > 0: a Givens rotation
    in the (a_i, a_j) plane by t_ij = rotation * g_i * l_j moves probability
    from j to i, limited, as in QditTS, so that a_j^2 stays at the floor.
kernel[k] is the weight at distance k (default 1, 0.5); weight outside the
domain is dropped before renormalizing.  The angles add up to `rotation`, so
kernel (1,) is exactly QditTS.  Pairs are applied one after another (gain
offset outer, loss offset inner), each one norm-preserving.  Both sensing
levels and priority ranks are ordered, so both are smoothed.  b == w is tabu.

Floor eps = min(probability_floor, 1/d): options already at the floor are
left out of l (l is renormalized over the rest), so a single option at the
floor does not stop the rest of the row.

History: before 2026-10-18 the register stored probabilities P and updated
P += delta * (g - l) with delta=0.02 (no rotation).
"""

import numpy as np

from Algorithm.quantum.QditTS import QditTS


class QditTS_Smooth(QditTS):
    def __init__(self, DP, n=10, rotation=0.01 * np.pi, kernel=(1.0, 0.5),
                 probability_floor=0.01, seed=None):
        kernel = np.asarray(kernel, dtype=float)
        if kernel.ndim != 1 or not len(kernel) or kernel[0] <= 0 or np.any(kernel < 0):
            raise ValueError("kernel must be nonnegative with kernel[0] > 0")
        self.kernel = kernel
        super().__init__(DP, n=n, rotation=rotation, probability_floor=probability_floor,
                         seed=seed)
        self.name = "QditTS_Smooth"

    def _centred(self, centre, mask):
        distance = np.abs(np.arange(self.valid.shape[1]) - centre[:, None])
        weight = np.where(distance < len(self.kernel),
                          self.kernel[np.minimum(distance, len(self.kernel) - 1)], 0.0)
        weight = weight * mask
        total = weight.sum(axis=1, keepdims=True)
        return np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)

    def _update(self, best_code, worst_code):
        best_code = np.asarray(best_code, dtype=int)
        worst_code = np.asarray(worst_code, dtype=int)
        rows = np.flatnonzero(best_code != worst_code)
        if not len(rows):
            return
        a = self.amplitudes[rows]
        floor = self.amplitude_floor[rows]
        best, worst = best_code[rows], worst_code[rows]
        gain = self._centred(best, self.valid[rows])
        loss = self._centred(worst, self.valid[rows] & (a > floor[:, None] + 1e-12))
        width = a.shape[1]
        index = np.arange(len(rows))
        offsets = np.arange(1 - len(self.kernel), len(self.kernel))
        for gain_offset in offsets:
            i = best + gain_offset
            for loss_offset in offsets:
                j = worst + loss_offset
                inside = (i >= 0) & (i < width) & (j >= 0) & (j < width) & (i != j)
                r, ri, rj = index[inside], i[inside], j[inside]
                weight = gain[r, ri] * loss[r, rj]
                keep = weight > 0
                r, ri, rj, weight = r[keep], ri[keep], rj[keep], weight[keep]
                if not len(r):
                    continue
                a_i, a_j = a[r, ri], a[r, rj]
                radius = np.hypot(a_i, a_j)
                limit = np.arctan2(a_j, a_i) - np.arcsin(np.minimum(1.0, floor[r] / radius))
                angle = np.clip(limit, 0.0, self.rotation * weight)
                cos, sin = np.cos(angle), np.sin(angle)
                a[r, ri] = a_i * cos + a_j * sin
                a[r, rj] = a_j * cos - a_i * sin
        self.amplitudes[rows] = a
