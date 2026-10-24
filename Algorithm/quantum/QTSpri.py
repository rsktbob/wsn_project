"""QTS on the PriorityEncoding chromosome (same chromosome as Ring_SEpri).

Same loop, iteration-best rotation, tabu gates, probability floor and uniform
prior as QTS; only the register changes to QuantumPriorityEncoding: every
sensor has an activation priority (energy E) and a routing priority (E plus
farness F), 10 values and 9 thermometer gates each, 1800 Q-bits for 100
sensors.  Sensing levels are chosen greedily by the PriorityEncoding decoder.

Defaults n=30, rotation=0.08*pi are the best iteration-best setting of the
2026-10-18 grid (MAP01-06, seed 7-11, budget 10000, v3: 1164.4 against
ring_sepri 1156.2, p=.075; seed 12-21: 1150.0 against 1158.9, p=.047).
"""

import numpy as np

from Algorithm.quantum.QTS import QTS
from State.QuantumPriorityEncoding import QuantumPriorityEncoding


class QTSpri(QTS):
    def __init__(self, DP, n=30, rotation=0.08 * np.pi, probability_floor=0.01,
                 init="uniform", seed=None):
        super().__init__(DP, n=n, rotation=rotation, probability_floor=probability_floor,
                         init=init, seed=seed)
        self.name = "QTSpri"

    def _build_encoding(self, problem):
        return QuantumPriorityEncoding(problem.SENSOR_NUMBER)


__all__ = ["QTSpri"]
