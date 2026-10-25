"""Finite algebra checks for the research report; not a quantum benchmark."""

import itertools
import json
import math
from pathlib import Path


ENCODINGS = {
    "binary": ((0, 0), (0, 1), (1, 0)),
    "gray": ((0, 0), (0, 1), (1, 1)),
    "thermometer": ((0, 0), (1, 0), (1, 1)),
}


def indicators(name, bits):
    a, b = bits
    if name == "binary":
        return (1 - a - b, b, a)
    if name == "gray":
        return (1 - b, b - a, a)
    return (1 - a, a - b, b)


def run():
    # The nine elementary 3x3 tables form a basis for every pairwise cost table.
    # Check every basis table on all nine legal level assignments, for every
    # combination of encodings on the two sensors, including mixed encodings.
    checks = 0
    for ni, ci in ENCODINGS.items():
        for nj, cj in ENCODINGS.items():
            for u, v, li, lj in itertools.product(range(3), repeat=4):
                encoded = indicators(ni, ci[li])[u] * indicators(nj, cj[lj])[v]
                expected = int(li == u and lj == v)
                assert encoded == expected
                checks += 1

    # Simple two-sensor example: costs (0,1,3), redundant activation cost 2,
    # coverage requires at least one active sensor, slack is 0 or 1.
    # E = cost_i + cost_j + 2*a_i*a_j + 20*invalid + 20*(a_i+a_j-1-s)^2.
    # Every a_i and every cost_i is affine in the two encoded bits, so this is
    # a QUBO after reducing b^2=b. Enumerate all 32 bitstrings per encoding.
    toy_results = {}
    for name, code in ENCODINGS.items():
        best = None
        minimizers = []
        for bits in itertools.product((0, 1), repeat=5):
            left, right, slack = bits[:2], bits[2:4], bits[4]
            vi, vj = indicators(name, left), indicators(name, right)
            ai, aj = vi[1] + vi[2], vj[1] + vj[2]
            invalid = int(left not in code) + int(right not in code)
            energy = (vi[1] + 3 * vi[2] + vj[1] + 3 * vj[2]
                      + 2 * ai * aj + 20 * invalid
                      + 20 * (ai + aj - 1 - slack) ** 2)
            if best is None or energy < best:
                best, minimizers = energy, [bits]
            elif energy == best:
                minimizers.append(bits)
        decoded = sorted((code.index(x[:2]), code.index(x[2:4])) for x in minimizers)
        assert best == 1 and decoded == [(0, 1), (1, 0)]
        toy_results[name] = {"qubo_minimum": best, "sensor_levels": decoded,
                             "enumerated_bitstrings": 32}

    return {
        "scope": "finite algebra verification; no quantum circuits or WSN benchmark",
        "pairwise_basis_checks": checks,
        "pairwise_basis_failures": 0,
        "toy_qubo_checks": toy_results,
        "uniform_valid_binary_probability_100": (3 / 4) ** 100,
        "global_ternary_basis_qubits_100": (3 ** 100 - 1).bit_length(),
        "amplitude_features_100_qubits": math.ceil(math.log2(100)),
        "qrao_ideal_lower_bound_200": math.ceil(200 / 3),
        "pce_pair_slots": {str(n): 3 * math.comb(n, 2) for n in (12, 13)},
        "statevector_gib": {str(n): 16 * 2 ** n / 2 ** 30 for n in (13, 28, 29, 30, 31)},
        "density_matrix_gib": {str(n): 16 * 4 ** n / 2 ** 30 for n in (13, 14, 15, 16)},
        "qutrit_statevector_gib": {str(n): 16 * 3 ** n / 2 ** 30 for n in (17, 18, 19, 20)},
    }


if __name__ == "__main__":
    output = Path(__file__).with_suffix(".json")
    result = run()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
