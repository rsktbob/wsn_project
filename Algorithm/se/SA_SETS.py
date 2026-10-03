"""Adaptive identity selection and CCS initialization on the SETS market."""
from Algorithm.se.SETS import SETS
from Algorithm.se.sensor_operators import AdaptiveSensorInitialization


class SA_SETS(AdaptiveSensorInitialization, SETS):
    pass


__all__ = ["SA_SETS"]
