"""Read-only WSN timing audit; writes only its JSON result under artifacts.

Run using the existing wsnenv. Timings exclude imports, Problem construction,
candidate generation and JIT warmup. No optimizer-result cache is used.
"""
import contextlib
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "codes" / "2027WSN"
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
import numpy as np
import numba
from Problem.Problem import Problem
from Problem.services.fitness_servicev2 import FitnessServiceV2
from Problem.services.evaluation_kernels import NUMBA_AVAILABLE
from State.SensorEncoding import SensorEncoding, DEFAULT_SENSOR_ENCODING

COUNT = 256
REPEATS = 5
SEED = 20261001


def timing(items, function):
    batches = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        for item in items:
            function(item)
        batches.append((time.perf_counter_ns() - start) / len(items) / 1e6)
    start = time.perf_counter_ns()
    samples = []
    for item in items:
        t = time.perf_counter_ns()
        function(item)
        samples.append((time.perf_counter_ns() - t) / 1e6)
    return {
        "batch_mean_ms_per_candidate": batches,
        "median_batch_mean_ms": float(np.median(batches)),
        "individual_median_ms": float(np.median(samples)),
        "individual_p95_ms": float(np.percentile(samples, 95)),
    }


cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
result = {
    "date": "2026-10-01", "cpu": cpu, "python": sys.version,
    "platform": platform.platform(), "numpy": np.__version__,
    "numba": numba.__version__, "numba_available": NUMBA_AVAILABLE,
    "decoder": DEFAULT_SENSOR_ENCODING, "fitness": "v2", "seed": SEED,
    "candidates_per_case": COUNT, "batch_repeats": REPEATS,
    "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    "note": "Initial energy=10; random chromosomes, not optimized population; no lifetime rollout or movement. Three-level sensitivity uses physical radii [0,7.5,15], not a claimed user setting. Evaluation-only reuses decoded states; combined creates fresh states.",
    "cases": [],
}
for radii_name, radii in [("default_six", None), ("illustrative_three", [0, 7.5, 15])]:
    for map_index in range(1, 7):
        random.seed(SEED + map_index)
        np.random.seed(SEED + map_index)
        rng = np.random.default_rng(SEED + map_index)
        with contextlib.redirect_stdout(io.StringIO()):
            p = Problem(B=100, S=100, T=100, F=10, FILE=f"MAP{map_index:02}", sensing_mode="discrete", sensing_levels=radii)
        p.fitness_service = FitnessServiceV2(p)
        candidates = [SensorEncoding.random(p.SENSOR_NUMBER, p.radius_option_counts, rng) for _ in range(COUNT)]
        before_codes = [c.code.copy() for c in candidates]
        energy_before = p.energy.copy()
        def decode(c):
            return c.decode(p)
        def score(s):
            return float(np.sum(p.evaluate_state(s)))
        def combined(c):
            return score(decode(c))
        states = [decode(c) for c in candidates]
        scores = np.array([score(s) for s in states])
        for c in candidates[:32]:
            combined(c)
        row = {
            "map": f"MAP{map_index:02}", "radii_case": radii_name,
            "physical_radii": p.DISCRETE_LEVEL_RANGE.tolist(),
            "sensor_count": p.SENSOR_NUMBER, "target_count": p.TARGET_NUMBER,
            "feasible_count": int(np.sum(scores >= 0)),
            "incomplete_count": int(np.sum((scores < 0) & (scores > -2))),
            "invalid_count": int(np.sum(scores <= -2)),
            "decode_and_route": timing(candidates, decode),
            "fitness_on_decoded_state": timing(states, score),
            "decode_route_and_fitness": timing(candidates, combined),
        }
        np.testing.assert_array_equal(p.energy, energy_before)
        for c, before in zip(candidates, before_codes):
            np.testing.assert_array_equal(c.code, before)
        np.testing.assert_allclose([combined(c) for c in candidates], scores, rtol=0, atol=1e-12)
        result["cases"].append(row)
        print(json.dumps({"map": row["map"], "levels": radii_name, "feasible": row["feasible_count"], "fitness_ms": row["fitness_on_decoded_state"]["median_batch_mean_ms"], "combined_ms": row["decode_route_and_fitness"]["median_batch_mean_ms"]}), flush=True)
result["source_hashes"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), PROJECT / "State/SensorEncoding.py", PROJECT / "Problem/services/fitness_servicev2.py", PROJECT / "Problem/services/energy_service.py", PROJECT / "Problem/services/routing_service.py", PROJECT / "Problem/services/coverage_service.py", PROJECT / "Problem/services/evaluation_kernels.py"]}
out = ROOT / "artifacts" / "benchmark_fitness_20261001.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(f"Saved {out}", flush=True)
