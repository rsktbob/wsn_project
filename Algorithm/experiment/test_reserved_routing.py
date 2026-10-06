import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Algorithm.experiment.reserved_routing import ReservedRouting, Offer, choose_offer


def problem():
    caps = np.zeros((3, 2, 4))
    caps[0, 1, 3] = 10
    caps[1, 1, 0] = 20
    caps[2, 1, 0] = 20
    return SimpleNamespace(SENSOR_NUMBER=3, DEVICE_NUMBER=4, BSID=3,
                           radius_option_counts=np.full(3, 2),
                           device=np.array([[1.,0.],[2.,0.],[3.,0.],[0.,0.]]), BS=np.array([0.,0.]),
                           amp_factors=[10.,.0013], d0=87., circuit_cost=5e-6,
                           generated_load=np.array([2,3,6]), link_capacity=caps,
                           energy=np.ones(3), sensing_costs=np.full((3,2), .001), amp_cost=np.zeros((3,4)),
                           resolve_state_radius=lambda s: None)


class Tests(unittest.TestCase):
    def test_local_offer(self):
        self.assertEqual(choose_offer([Offer(0, 100, 4), Offer(1, 6, 30)], 5).parent, 1)
        self.assertIsNone(choose_offer([Offer(0, 100, 4)], 5))

    def test_lifetime_precedes_packet_margin(self):
        offers = [Offer(0, 100, 100, 2, 10), Offer(1, 10, 10, 5, 5)]
        self.assertEqual(choose_offer(offers, 1).parent, 1)

    def test_reservation_prevents_shared_upstream_overload(self):
        p = problem()
        caps = p.link_capacity.copy()
        result = ReservedRouting().build(p, np.ones(3, dtype=int))
        self.assertEqual(result.state.next_hops.tolist(), [3, 0, -1])
        self.assertEqual(result.state.tx_load.tolist(), [5, 3, 0])
        self.assertTrue(np.all(result.control > 0))
        self.assertGreater(result.messages, 0)
        np.testing.assert_array_equal(p.link_capacity, caps)
        np.testing.assert_array_equal(result.state.levels, [1,1,1])

    def test_control_ablation_same_routes_and_off_nodes(self):
        p = problem()
        levels = np.array([1,0,1])
        paid = ReservedRouting(128).build(p, levels)
        free = ReservedRouting(0).build(p, levels)
        np.testing.assert_array_equal(paid.state.next_hops, free.state.next_hops)
        self.assertEqual(paid.state.next_hops.tolist(), [3,-1,0])
        self.assertEqual(paid.state.tx_load.tolist(), [8,0,6])
        self.assertEqual(float(paid.control[1]), 0.)
        self.assertEqual(float(free.control.sum()), 0.)

    def test_cycle_free_and_same_schedule(self):
        p = problem()
        p.link_capacity[:,1,:] = 100
        r = ReservedRouting().build(p, np.ones(3,dtype=int))
        for start in range(3):
            node, seen = start, set()
            while node != p.BSID:
                self.assertNotIn(node, seen)
                seen.add(node)
                node = int(r.state.next_hops[node])
                self.assertGreaterEqual(node, 0)
        np.testing.assert_array_equal(r.state.levels, [1,1,1])

if __name__ == '__main__':
    unittest.main()
