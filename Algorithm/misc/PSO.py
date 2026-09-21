import numpy as np

from Algorithm.misc.BasePSO import BasePSO
from State.SensorEncoding import SensorEncoding


class PSO(BasePSO):
    """PSO over continuous positions decoded through ``SensorEncoding``."""

    def __init__(
        self, P, n=50, w=0.7968, end_w=0.4, vmax=0.5, vmin=-0.5, fmax=1.0, fmin=0
    ):
        super().__init__(n, w, end_w, vmax, vmin, fmax, fmin)
        self.gene_domain = max(P.LEVEL, SensorEncoding.RANK_PRECISION)
        P.prepare_coding_cache()

    def _create_state(self, P, sch_s):
        code = np.random.uniform(
            0,
            self.fmax,
            P.SENSOR_NUMBER * 2,
        )
        return SensorEncoding(code)

    def _decode_particle(self, P, coding):
        discrete = SensorEncoding(self._discretize_code(coding))
        return discrete.decode(P)

    def _discretize_code(self, state):
        span = self.fmax - self.fmin
        if span <= 0:
            span = 1.0
        normalized = (np.clip(state.code, self.fmin, self.fmax) - self.fmin) / span
        code = np.floor(normalized * self.gene_domain).astype("int")
        return np.clip(code, 0, self.gene_domain - 1)

    def _to_discrete_state(self, P, coding):
        if np.issubdtype(np.asarray(coding.code).dtype, np.floating):
            code = self._discretize_code(coding)
        else:
            code = np.asarray(coding.code).astype("int")
        return SensorEncoding(code)

    def _evaluate_state(self, P, coding):
        if np.issubdtype(np.asarray(coding.code).dtype, np.floating):
            state = self._decode_particle(P, coding)
        else:
            state = coding.decode(P)
        self.evatime += 1
        return P.evaluate_state(state)

    def search(self, problem, budget, state=None):
        best_state = self._optimize(problem, budget, state)
        if best_state is None:
            return None
        return self._to_discrete_state(problem, best_state).decode(problem)
