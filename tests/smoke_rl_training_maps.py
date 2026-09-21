"""Audit the real map cohort and generator reproducibility without training."""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CreateMapData import generate_map, geometry_fingerprint
from train_rl_sets import parse_args, build_training_problem, audit_training_maps


def main():
    args = parse_args([])
    assert len(args.map_specs) == 200 and len(args.validation_specs) == 40
    fingerprints = set()
    coverage_min = []
    with contextlib.redirect_stdout(io.StringIO()):
        for spec in args.map_specs + args.validation_specs:
            problem = build_training_problem(args, 7, spec)
            assert problem.sensor.shape == (100, 2)
            assert problem.target.shape == (100, 2)
            assert problem.BOUNDARY == 100 and np.all(problem.initial_energy == 10)
            assert problem.validate_problem()
            assert geometry_fingerprint(problem.sensor) == spec["geometry_sha256"]
            fingerprints.add(spec["geometry_sha256"])
            coverage_min.append(int(np.min(np.sum(problem.target_sensor_mask, axis=1))))
        with tempfile.TemporaryDirectory() as temporary:
            regenerated = generate_map(100, 100, 100, "REPRO", 20001, temporary)
            metadata = json.loads((regenerated / "metadata.json").read_text())
            assert metadata["geometry_sha256"] == args.map_specs[0]["geometry_sha256"]
            try:
                generate_map(100, 100, 100, "REPRO", 20001, temporary)
            except FileExistsError:
                pass
            else:
                raise AssertionError("generator overwrote an existing dataset")
            generate_map(100, 100, 100, "REPRO", 20001, temporary, overwrite=True)
            replaced = json.loads((regenerated / "metadata.json").read_text())
            assert replaced["coverage_candidates_min"] >= 3
    assert len(fingerprints) == 240
    try:
        audit_training_maps(args.map_specs, [args.map_specs[0]])
    except ValueError:
        pass
    else:
        raise AssertionError("train/validation overlap accepted")
    assert min(coverage_min) >= 3
    print("smoke_rl_training_maps_ok", "240 unique maps; minimum coverage candidates:", min(coverage_min))


if __name__ == "__main__":
    main()
