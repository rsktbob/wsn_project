"""Conditional-thermometer Q-bits for PriorityEncoding (used by QTSpri).

Same gates, observation and elite path as QuantumSensorEncoding, but every
gene is a priority with PriorityEncoding.RANK_PRECISION (10) values: each
sensor has an activation gene a and a routing gene b, [a0, b0, a1, b1, ...],
9 gates each, so 100 sensors give 1800 Q-bits.  observe() returns a
PriorityEncoding candidate; the decoder chooses the sensing levels.
"""

import numpy as np

from State.PriorityEncoding import PriorityEncoding
from State.QuantumSensorEncoding import QuantumSensorEncoding


class QuantumPriorityEncoding(QuantumSensorEncoding):
    def __init__(self, sensor_count):
        # Interleaved [a, b] with 10 values each is the thermometer layout of
        # "10 sensing options + 10 priority ranks" per sensor.
        super().__init__(np.full(int(sensor_count), PriorityEncoding.RANK_PRECISION))

    def observe(self, probabilities, rng):
        candidate, bits = super().observe(probabilities, rng)
        return PriorityEncoding(candidate.code), bits
