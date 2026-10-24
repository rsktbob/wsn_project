"""Qdit-TS on the PriorityEncoding chromosome (same chromosome as Ring_SEpri).

Same loop, Givens rotation, tabu, probability floor and uniform prior as
QditTS; only the register changes: every sensor has an activation priority
(energy E) and a routing priority (E plus farness F), one row of 10
amplitudes each, [a0, b0, a1, b1, ...], 2000 amplitudes for 100 sensors.
observe() returns a PriorityEncoding candidate; the decoder chooses the
sensing levels greedily.

Defaults n=10, rotation=0.03*pi are the best setting of the 2026-10-18 grid
(iteration best, MAP01-06, seed 7-11, budget 10000, v3: 1173.3).  On seed
12-21: 1164.3 against ring_sepri 1158.9 (+0.5%, p=.11) and QTSpri 1150.0
(+1.2%, p=.002).
"""

import numpy as np

from Algorithm.quantum.QditTS import QditTS
from State.PriorityEncoding import PriorityEncoding


class QditTSpri(QditTS):
    def __init__(self, DP, n=10, rotation=0.03 * np.pi, probability_floor=0.01,
                 seed=None):
        super().__init__(DP, n=n, rotation=rotation, probability_floor=probability_floor,
                         seed=seed)
        self.name = "QditTSpri"

    def _configure(self, sensing_domains):
        # Interleaved [a, b] with 10 values each: the QditTS layout with
        # 10 options in place of the sensing levels.
        super()._configure(np.full(len(sensing_domains), PriorityEncoding.RANK_PRECISION))

    def observe(self):
        return PriorityEncoding(super().observe().code)


__all__ = ["QditTSpri"]
