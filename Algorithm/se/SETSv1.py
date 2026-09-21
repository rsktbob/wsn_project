"""First retained SETS variant using the adaptive SA-SETS market flow."""

from __future__ import annotations

from Algorithm.se.SA_SETS import SA_SETS
from State.SensorEncoding import SensorEncoding


class SETSv1(SA_SETS):
    """Use uniform sensor coding and the first live identity sensors.

    SETSv1 and SA-SETS share the same regional transition, Beta probability,
    adaptive belief update, and SE loop.  They differ only in initialization
    and identity-sensor selection, so this class overrides exactly those rules.
    """

    def select_identity_sensors(self, problem):
        return [
            sensor_id
            for sensor_id in range(problem.SENSOR_NUMBER)
            if problem.energy[sensor_id] >= problem.liveJ
        ][: self.identity_bit_count]

    def create_candidate(self, problem, region=None):
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate


__all__ = ["SETSv1"]
