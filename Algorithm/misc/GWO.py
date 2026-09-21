import numpy as np

from Algorithm.misc.BaseGWO import BaseGWO
from State.SensorEncoding import SensorEncoding


class GWO(BaseGWO):
    """GWO over continuous positions decoded through ``SensorEncoding``."""

    def __init__(self, P, n=10):
        super().__init__(P, n, 0, 1)
        self.gene_domain = max(P.LEVEL, SensorEncoding.RANK_PRECISION)
        P.prepare_coding_cache()

    def _decode_state(self, P, coding):
        code = np.floor(coding.code * self.gene_domain).astype("int")
        discrete = SensorEncoding(code)
        return discrete.decode(P)

    def _create_state(self, P, sch_s):
        coding = SensorEncoding.random(P.SENSOR_NUMBER, self.gene_domain)
        coding.code = coding.code / self.gene_domain
        return coding

    def _evaluate_state(self, P, coding, DEBUG=False):
        self.evatime += 1
        state = self._decode_state(P, coding)
        return P.evaluate_state(state)

    def _to_discrete_state(self, P, coding):
        code = np.floor(
            np.clip(coding.code, self.lb, self.ub) * self.gene_domain
        ).astype(int)
        code = np.clip(
            code,
            0,
            self.gene_domain - 1,
        )
        return SensorEncoding(code)

    def search(self, problem, budget, state=None):
        best_state = self._optimize(problem, budget, state)
        if best_state is None:
            return None
        return self._to_discrete_state(problem, best_state).decode(problem)
