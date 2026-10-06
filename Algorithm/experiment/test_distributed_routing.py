"""Protocol invariants, local-policy decisions, and physical cost checks."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Algorithm.experiment.distributed_routing import (
    Advertisement, Neighbor, LocalObservation, LocalRoutingPolicy, DistributedRouting,
)


def toy_problem():
    # 0 -> sink, 1 -> 0, 2 -> 1; unequal source loads catch accumulation bugs.
    device = np.array([[1., 0.], [2., 0.], [3., 0.], [0., 0.]])
    distances = np.linalg.norm(device[:, None] - device[None, :], axis=2)
    np.fill_diagonal(distances, 1e9)
    return SimpleNamespace(SENSOR_NUMBER=3, BSID=3, DEVICE_NUMBER=4,
                           device=device, distances=distances, blocked_distance=1e9,
                           radius_option_counts=np.full(3, 2), energy=np.full(3, 100.),
                           sensing_costs=np.zeros((3, 2)), generated_load=np.array([2, 3, 5]),
                           d0=87., amp_factors=[10., .0013], circuit_cost=5e-6)


class ProtocolTests(unittest.TestCase):
    def test_energy_changes_local_choice(self):
        weak = Neighbor(Advertisement(0, 1., 1., True, .2, 1), .01)
        strong = Neighbor(Advertisement(1, 2., 100., True, .001, 1), .02)
        equal_rank = Neighbor(Advertisement(2, 3., 100., True, 0., 1), 0.)
        o = LocalObservation(3, 3., 100., .01, 4, (weak, strong, equal_rank))
        self.assertEqual(LocalRoutingPolicy().choose(o)[2], 1)
        self.assertIsNone(LocalRoutingPolicy().choose(LocalObservation(3, 3., 0., .01, 4, o.neighbors)))

    def test_chain_load_rank_messages_and_no_mutation(self):
        p = toy_problem()
        before = p.energy.copy()
        epoch = DistributedRouting(radio_range=1.1).build(p, np.ones(3, dtype=int))
        self.assertEqual(epoch.state.next_hops.tolist(), [3, 0, 1])
        self.assertEqual(epoch.state.tx_load.tolist(), [10, 8, 5])
        self.assertEqual(epoch.state.paths[2], [2, 1, 0, 3])
        self.assertEqual(epoch.waves, 3)
        self.assertEqual(epoch.control_messages, 12)
        # node 1 has two peers, two sends + two receives per peer.
        expected = 4 * 128 * (2 * p.circuit_cost + 10e-9)
        self.assertAlmostEqual(epoch.control_cost[1], expected)
        np.testing.assert_array_equal(p.energy, before)
        self.assertFalse(epoch.disconnected_ids)

    def test_colocated_sink_is_reachable(self):
        p = toy_problem()
        p.device[0] = p.device[p.BSID]
        epoch = DistributedRouting(radio_range=1.1).build(p, np.ones(3, dtype=int))
        self.assertEqual(int(epoch.state.next_hops[0]), p.BSID)

    def test_lifetime_energy_and_cap(self):
        from Algorithm.experiment.run_distributed_routing import simulate
        p = toy_problem()
        p.find_uncovered_targets = lambda state: []
        p.energy_service = SimpleNamespace(calculate_total_cost=lambda state: np.full(3, 40.))
        router = DistributedRouting(radio_range=1.1, control_bits=0)
        summary, trace = simulate(p, router, np.ones(3, dtype=int), 10)
        self.assertEqual(summary["completed_slots"], 2)
        self.assertEqual(summary["stop_reason"], "data_energy_shortfall")
        self.assertFalse(summary["truncated"])
        np.testing.assert_array_equal(p.energy, np.full(3, 20.))
        p.energy[:] = 100.
        summary, trace = simulate(p, router, np.ones(3, dtype=int), 1)
        self.assertEqual(summary["completed_slots"], 1)
        self.assertTrue(summary["truncated"])

    def test_partition_fails_explicitly(self):
        epoch = DistributedRouting(radio_range=.5).build(toy_problem(), np.ones(3, dtype=int))
        self.assertEqual(epoch.disconnected_ids, [0, 1, 2])
        self.assertTrue(np.all(epoch.state.next_hops == -1))

    def test_control_energy_blocks_route(self):
        p = toy_problem()
        p.energy[0] = 1e-10
        epoch = DistributedRouting(radio_range=1.1).build(p, np.ones(3, dtype=int))
        self.assertEqual(epoch.disconnected_ids, [0, 1, 2])

    def test_illegal_level_and_repeatability(self):
        p = toy_problem()
        router = DistributedRouting(radio_range=1.1)
        for levels in ([0, 1, 1], [2, 1, 1], [1., 1., 1.]):
            with self.assertRaises(ValueError):
                router.build(p, levels)
        a = router.build(p, np.ones(3, dtype=int))
        b = router.build(p, np.ones(3, dtype=int))
        np.testing.assert_array_equal(a.state.next_hops, b.state.next_hops)
        free = DistributedRouting(radio_range=1.1, control_bits=0).build(p, np.ones(3, dtype=int))
        self.assertEqual(float(free.control_cost.sum()), 0.)


if __name__ == '__main__':
    unittest.main()
