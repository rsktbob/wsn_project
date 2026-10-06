import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['NUMBA_CACHE_DIR']=str(HERE/'.numba_cache')
sys.path.insert(0,str(HERE.parents[1]))
from Algorithm.experiment.quantum_local_v2 import LevelModel,LocalLevelPolicy,LocalFeedback,QuantumLocalV2
from Algorithm.experiment.local_scheduling import SchedulingView,LocalScheduleRouting


def view(node=0,current=0):
    return SchedulingView(node,10.,10.,current,
            tuple(frozenset(range(k)) for k in range(6)),tuple(.001*k*k for k in range(6)),.002,frozenset())


class Tests(unittest.TestCase):
    def test_uniform_categorical_prior_and_single_option(self):
        m=LevelModel(5)
        np.testing.assert_allclose(m.masses(),np.full(5,.2))
        self.assertEqual(m.angles.shape,(4,))
        np.testing.assert_allclose(np.sin(m.angles)**2,m.probabilities)
        single=LevelModel(1);single.update(0,1)
        np.testing.assert_array_equal(single.masses(),[1.])

    def test_rotation_direction_and_inactive_suffix(self):
        m=LevelModel(5);before=m.masses()[2];suffix=m.angles[3]
        m.update(2,1)
        self.assertGreater(m.masses()[2],before)
        self.assertEqual(m.angles[3],suffix)
        m=LevelModel(5);before=m.masses()[2];m.update(2,0)
        self.assertLess(m.masses()[2],before)

    def test_bounds_and_classical_control(self):
        for mode in ('quantum','classical'):
            m=LevelModel(5,mode)
            for k in range(1000): m.update(k%5,float(k%2))
            self.assertTrue(np.all(m.probabilities>=m.floor-1e-12))
            self.assertTrue(np.all(m.probabilities<=1-m.floor+1e-12))
            self.assertAlmostEqual(float(m.masses().sum()),1.)
        m=LevelModel(5,'random');before=m.probabilities.copy();m.update(2,1)
        np.testing.assert_array_equal(m.probabilities,before)

    def test_same_initial_sampling_and_legal_mask(self):
        policies=[LocalLevelPolicy(mode,7,1.) for mode in ('quantum','classical','random')]
        sequences=[]
        for policy in policies:
            seq=[]
            for _ in range(30):
                policy.start_epoch(); c=policy.propose(view(current=3))
                self.assertIn(c.level,[4,5]);seq.append(c.level)
                self.assertEqual(policy.propose(view(current=3)),c)
            sequences.append(seq)
        self.assertEqual(sequences[0],sequences[1]);self.assertEqual(sequences[0],sequences[2])

    def test_node_rng_independence_and_memory(self):
        a=LocalLevelPolicy(seed=7,exploration=1.);b=LocalLevelPolicy(seed=7,exploration=1.)
        ca=a.propose(view(0));a.propose(view(1))
        b.propose(view(1));cb=b.propose(view(0));self.assertEqual(ca,cb)
        before=a.models[0].angles.copy()
        a.feedback(0,LocalFeedback(ca.level,True,10.,.01,0.))
        after=a.models[0].angles.copy();self.assertFalse(np.array_equal(before,after))
        a.start_epoch();np.testing.assert_array_equal(a.models[0].angles,after)

    def test_greedy_hook_matches_unchanged_original(self):
        from experiments.maps import load_map_specs
        from experiments.problem_setup import build_problem
        spec=load_map_specs('maps/maps_100100100.json')[0]
        p=build_problem(SimpleNamespace(_current_map=spec,sensing_mode='discrete',fitness_service='v2'),7)
        old=LocalScheduleRouting(True).build(p)
        new=QuantumLocalV2('greedy').build(p)
        for key in ('levels','next_hops','tx_load'):
            np.testing.assert_array_equal(getattr(old.state,key),getattr(new.state,key))
        np.testing.assert_array_equal(old.control,new.control)
        self.assertEqual(old.messages,new.messages)
        self.assertEqual(old.schedule.claims,new.schedule.claims)


if __name__=='__main__': unittest.main()
