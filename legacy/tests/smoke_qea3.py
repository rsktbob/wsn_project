"""Validate QEA3 migration semantics, ordering, budgets and shared decoding."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Algorithm.quantum.QEA3 import QEA3
from State.SensorEncoding import SensorEncoding
from Problem.Problem import Problem
from experiments.cli import parse_args
from experiments.builder import build_algorithm


def main():
    np.random.seed(11)
    p=Problem(B=50,S=30,T=9,F=100,FILE=None)
    a=QEA3(p,n=5,global_period=2,seed=7)
    a.run(p,budget=5)
    initial=a.angles.copy()
    np.testing.assert_array_equal(initial,np.tile(a.encoding.initial_angles,(5,1)))
    a.reference_scores=np.array([1.,3.,4.,2.,5.])
    before=[r.copy() for r in a.references]
    a._migrate(1)
    for index,winner in enumerate((1,1,2,2,4)):
        np.testing.assert_array_equal(a.references[index].code,before[winner].code)
    assert a.references[0] is not a.references[1]
    assert not np.shares_memory(a.references[0].code,a.references[1].code)
    np.testing.assert_array_equal(a.reference_scores,[3,3,4,4,5])
    np.testing.assert_array_equal(a.angles,initial)
    a._migrate(2)
    for ref in a.references:
        np.testing.assert_array_equal(ref.code,a.best_candidate.code)
        assert ref is not a.best_candidate
    np.testing.assert_array_equal(a.reference_scores,np.full(5,a.fitness))
    np.testing.assert_array_equal(a.angles,initial)
    assert a.local_migrations==1 and a.global_migrations==1

    class OrderingProbe(QEA3):
        def score(self, problem, candidate):
            super().score(problem,candidate)
            if self.evatime<=self.n:
                self.initial_refs.append(candidate.copy())
            # Force later candidates to improve references, so a mistaken
            # replace-before-rotate order cannot pass by chance.
            return float(self.evatime)
        def _rotate_references(self, observations):
            for ref,old in zip(self.references,self.initial_refs):
                np.testing.assert_array_equal(ref.code,old.code)
            super()._rotate_references(observations)
            self.checked=True
    probe=OrderingProbe(p,n=4,seed=7)
    probe.initial_refs=[]
    probe.checked=False
    probe.run(p,budget=8)
    assert probe.checked

    for mode in ('discrete','bucketed','exact'):
        np.random.seed(11)
        p=Problem(B=50,S=30,T=9,F=100,FILE=None,sensing_mode=mode)
        for budget in (1,4,5,7,20,23):
            a=QEA3(p,n=4,global_period=2,seed=77)
            r=a.run(p,budget=budget)
            assert r.evaluations==budget==len(r.history)
            assert np.all(np.diff(r.history)>=0)
            assert type(a.best_candidate) is SensorEncoding
            decoded=a.best_candidate.copy().decode(p)
            np.testing.assert_array_equal(decoded.levels,r.best_state.levels)
            np.testing.assert_array_equal(decoded.next_hops,r.best_state.next_hops)
            generations=max(0,(budget-4)//4)
            assert a.global_migrations==generations//2
            assert a.local_migrations==generations-generations//2
            repeat=QEA3(p,n=4,global_period=2,seed=77).run(p,budget=budget)
            np.testing.assert_array_equal(r.history,repeat.history)
            np.testing.assert_array_equal(r.best_state.next_hops,repeat.best_state.next_hops)
        a=QEA3(p,n=4,seed=2)
        r=a.run(p,budget=50,max_iteration=2)
        assert r.evaluations==12 and r.metadata['early_stopped']
        r=a.run(p,budget=1)
        assert r.evaluations==1 and a.local_migrations==a.global_migrations==0
    args=parse_args(['--algorithm','qea3','--evaluate','25'])
    assert isinstance(build_algorithm('qea3',p,args,7),QEA3)
    for period in (0,-1,1.5,float('nan')):
        try: QEA3(p,global_period=period)
        except ValueError: pass
        else: raise AssertionError(period)
    print('smoke_qea3_ok')

if __name__=='__main__': main()
