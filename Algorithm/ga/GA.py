import weakref
from Algorithm.ga.BaseGA import BaseGA
from State.SensorEncoding import SensorEncoding


class GA(BaseGA):
    def __init__(self, DP, n=50, cu=0.9, mu=0.1, seed=None):
        l = DP.SENSOR_NUMBER * 2
        super().__init__(n, l, cu, mu, seed=seed)
        self._decoded_snapshots = weakref.WeakKeyDictionary()
        DP.prepare_coding_cache()


    def _create_state(self, P, sch_s=None):
        coding = SensorEncoding.random(
            P.SENSOR_NUMBER,
            P.radius_option_counts,
            rng=self.rng,
        )
        # Seed coverage exactly as SA-SETS does, without its region/identity
        # constraints: every target contributes one valid (sensor, level)
        # choice from the current coding cache.  Later targets may overwrite
        # an earlier choice for the same sensor, which is intentional and
        # matches SA-SETS' historical initialization rule.
        for target_candidates in P.cover_candidates:
            if target_candidates:
                sensor_id, level = self.random.choice(target_candidates)
                coding.code[int(sensor_id) * 2] = int(level)

        # Preserve the mutation-time decoded snapshot used by this GA.
        self._decoded_snapshots[coding] = coding.decode(P)
        return coding

    def decode_candidate(self, P, candidate):
        """Reuse GA's mutation-time decoded snapshot for common evaluation."""
        decoded_state = self._decoded_snapshots.get(candidate)
        if decoded_state is None:
            decoded_state = candidate.decode(P)
            self._decoded_snapshots[candidate] = decoded_state
        return decoded_state

    def _copy_state(self, state):
        copied = state.copy()
        decoded_state = self._decoded_snapshots.get(state)
        if decoded_state is not None:
            self._decoded_snapshots[copied] = decoded_state.copy()
        return copied

    def _crossover_states(self, f, m, index1, index2):
        f.code, m.code = self.crossover_segment(f.code, m.code, index1, index2)
        return f, m

    def _mutate_state(self, P, vc):
        gene_id = self.random.randrange(vc.length)
        current_value = vc.code[gene_id]
        upper_bound = max(P.LEVEL, SensorEncoding.RANK_PRECISION)
        next_value = self.random.randrange(upper_bound)
        retry_count = 0
        while next_value == current_value and retry_count < 5:
            next_value = self.random.randrange(upper_bound)
            retry_count += 1
        vc.code[gene_id] = next_value
        # Preserve the former mutation-time routing repair side effect.
        self._decoded_snapshots[vc] = vc.decode(P)
        return vc
