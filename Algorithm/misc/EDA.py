from Algorithm.misc.BaseEDA import BaseEDA
from State.SensorEncoding import SensorEncoding


class EDA(BaseEDA):
    """EDA that searches sensor-oriented WSN chromosomes."""

    def __init__(self, P, n=50, alpha=0.8):
        gene_domain = max(P.LEVEL, SensorEncoding.RANK_PRECISION)
        super().__init__(n, gene_domain, P.SENSOR_NUMBER * 2, alpha)

    def _evaluate_state(self, P, coding):
        state = coding.decode(P)
        self.evatime += 1
        return P.evaluate_state(state)

    def _build_states(self, P, code_population):
        codings = []
        for code in code_population:
            codings.append(SensorEncoding(code))
        return codings
