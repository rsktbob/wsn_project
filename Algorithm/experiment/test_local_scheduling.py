import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Algorithm.experiment.local_scheduling import SchedulingView, propose, LocalScheduling


def toy():
    table = np.zeros((3, 3, 3), dtype=int)
    table[0,0,1:] = 1
    table[1,0,2] = 1
    table[1,1,1:] = 1
    table[2,1,2] = 1
    table[2,2,1:] = 1
    return SimpleNamespace(SENSOR_NUMBER=3, BSID=3, radius_option_counts=np.full(3,3),
         coverage_table=table, energy=np.full(3,10.), initial_energy=10.,
         generated_load=np.full(3,400), circuit_cost=5e-6, amp_cost=np.zeros((3,4)),
         sensing_costs=np.array([[0,.001,.004]]*3),
         device=np.array([[1.,0.],[2.,0.],[3.,0.],[0.,0.]]), d0=87.,amp_factors=[10,.0013])


class Tests(unittest.TestCase):
    def test_own_policy_and_received_claims(self):
        coverage=(frozenset(),frozenset([0]),frozenset([0,1]))
        v=SchedulingView(0,10,10,0,coverage,(0,.001,.004),.002,frozenset())
        self.assertIsNotNone(propose(v))
        v=SchedulingView(0,10,10,0,coverage,(0,.001,.004),.002,frozenset([0,1]))
        self.assertIsNone(propose(v))
        v=SchedulingView(0,0,10,0,coverage,(0,.001,.004),.002,frozenset())
        self.assertIsNone(propose(v))

    def test_unique_ownership_coverage_and_energy_unchanged(self):
        p=toy(); before=p.energy.copy()
        result=LocalScheduling().build(p)
        assigned=[]
        for c in result.claims:
            assigned.extend(c['targets'])
            self.assertTrue(all(p.coverage_table[t,c['node'],c['level']] for t in c['targets']))
        self.assertEqual(sorted(assigned),[0,1,2])
        self.assertTrue(np.all(p.coverage_table[:,np.arange(3),result.levels].sum(axis=1)>0))
        self.assertTrue(np.all(result.control>0))
        np.testing.assert_array_equal(p.energy,before)

    def test_low_energy_node_sleeps_without_control_charges(self):
        p=toy(); p.energy[0]=1e-8
        result=LocalScheduling().build(p)
        self.assertEqual(int(result.levels[0]),0)
        self.assertEqual(float(result.control[0]),0.)

    def test_off_is_allowed_and_no_invented_coverage(self):
        p=toy(); p.energy[:]=0
        result=LocalScheduling().build(p)
        self.assertTrue(np.all(result.levels==0))
        self.assertFalse(result.claims)
        self.assertEqual(result.messages,0)

    def test_reproducible_and_control_ablation(self):
        p=toy()
        a=LocalScheduling().build(p); b=LocalScheduling().build(p)
        np.testing.assert_array_equal(a.levels,b.levels)
        self.assertEqual(a.claims,b.claims)
        free=LocalScheduling(False).build(p)
        self.assertEqual(float(free.control.sum()),0.)
        self.assertGreater(free.messages,0)
        np.testing.assert_array_equal(a.levels,free.levels)
        self.assertEqual(a.claims,free.claims)
        import json
        json.dumps(a.claims)

if __name__=='__main__':
    unittest.main()
