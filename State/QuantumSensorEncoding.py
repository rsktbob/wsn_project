"""Conditional probability representation used by QEA.

This class configures and samples QEA probability gates. It is distinct from
SensorEncoding: observe() produces a SensorEncoding candidate, whose shared
decode() builds the WSN schedule and routing. There is no separate decoder here.
"""

import numpy as np

from State.SensorEncoding import SensorEncoding


class QuantumSensorEncoding:
    def __init__(self, sensing_domains):
        """Configure QEA sampling only; candidates use SensorEncoding."""
        sensing_domains = np.asarray(sensing_domains, dtype=int)
        if sensing_domains.ndim != 1 or not len(sensing_domains) or np.any(sensing_domains < 1):
            raise ValueError("sensing_domains must be a nonempty positive vector")
        self.domains = np.empty(2 * len(sensing_domains), dtype=int)
        self.domains[0::2] = sensing_domains
        self.domains[1::2] = SensorEncoding.RANK_PRECISION
        self.gene_ids = np.repeat(np.arange(len(self.domains)), self.domains - 1)
        self.offsets = np.concatenate([np.arange(k - 1) for k in self.domains])
        remaining = self.domains[self.gene_ids] - self.offsets
        self.initial_probabilities = (remaining - 1) / remaining
        self.initial_angles = np.arcsin(np.sqrt(self.initial_probabilities))
        self.length = len(self.gene_ids)

    def observe(self, probabilities, rng):
        """Sample a standard SensorEncoding, without an alternative decoder."""
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (self.length,):
            raise ValueError("probabilities have the wrong shape")
        bits = rng.random(self.length) < probabilities
        code = self.domains - 1
        np.minimum.at(code, self.gene_ids, np.where(bits, self.domains[self.gene_ids] - 1, self.offsets))
        return SensorEncoding(code), bits

    def elite_path(self, code):
        code = np.asarray(code, dtype=int)
        if code.shape != self.domains.shape or np.any(code < 0) or np.any(code >= self.domains):
            raise ValueError("elite code is outside the gene domains")
        values = code[self.gene_ids]
        # Gates after the first zero do not determine the observed category.
        return self.offsets < values, self.offsets <= values

