"""Generate reproducible WSN map datasets for experiments.

The experiment loader expects ``data/<B><S><T>_<ID>/sensor.csv``. Targets
remain on the deterministic grid used by :class:`Problem`, so a map seed
controls the randomly deployed sensor coordinates only.
"""

from __future__ import annotations

import argparse
import json
import random
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from Problem.Problem import Problem

PROJECT_ROOT = Path(__file__).resolve().parent


def geometry_fingerprint(sensor):
    """Identify coordinates independently of sensor ids and CSV formatting."""
    points = np.asarray(sensor, dtype="<f8")
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    return hashlib.sha256(points.tobytes()).hexdigest()


def dataset_fingerprint(spec):
    path = PROJECT_ROOT / "data" / (
        f'{spec["boundary"]}{spec["sensors"]}{spec["targets"]}_{spec["file"]}'
    ) / "sensor.csv"
    return geometry_fingerprint(pd.read_csv(path).values[:, 1:3])


def write_svg_preview(problem, output_path, dataset_name, seed):
    """Write a dependency-free SVG preview of sensors, targets, and the BS."""
    size = 720
    margin = 48
    scale = (size - 2 * margin) / float(problem.BOUNDARY)

    def point(x, y):
        return margin + float(x) * scale, size - margin - float(y) * scale

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="720" '
        'viewBox="0 0 720 720">',
        '<rect width="720" height="720" fill="white"/>',
        f'<text x="48" y="28" font-family="sans-serif" font-size="18">'
        f'{dataset_name} (map seed {seed})</text>',
        '<rect x="48" y="48" width="624" height="624" fill="none" '
        'stroke="#666" stroke-width="1"/>',
    ]
    for x, y in problem.target:
        sx, sy = point(x, y)
        elements.append(
            f'<path d="M {sx - 3:.2f} {sy - 3:.2f} L {sx + 3:.2f} {sy + 3:.2f} '
            f'M {sx - 3:.2f} {sy + 3:.2f} L {sx + 3:.2f} {sy - 3:.2f}" '
            'stroke="#d95f02" stroke-width="1.2"/>'
        )
    for x, y in problem.sensor:
        sx, sy = point(x, y)
        elements.append(
            f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="2.4" fill="#1b77b3"/>'
        )
    bsx, bsy = point(*problem.BS)
    elements.append(
        f'<circle cx="{bsx:.2f}" cy="{bsy:.2f}" r="6" fill="#2b8c3e" '
        'stroke="#145522" stroke-width="2"/>'
    )
    elements.append('</svg>')
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def generate_map(boundary, sensors, targets, map_id, seed, output_root, overwrite=False):
    """Generate one valid map and save its data, preview, and metadata."""
    if boundary <= 6 or not 0 < sensors <= (boundary - 6) ** 2 or targets <= 0:
        raise ValueError("invalid map size for the integer coordinate generator")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", map_id):
        raise ValueError("map id must contain only letters, numbers, _ or -")
    dataset_name = f"{boundary}{sensors}{targets}_{map_id}"
    dataset_dir = Path(output_root) / dataset_name
    if dataset_dir.exists() and not overwrite:
        raise FileExistsError(f"dataset already exists: {dataset_dir}")
    if dataset_dir.exists() and not dataset_dir.is_dir():
        raise FileExistsError(f"dataset path is not a directory: {dataset_dir}")
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    problem = Problem(
        B=int(boundary),
        S=int(sensors),
        T=int(targets),
        F=10,
        FILE=None,
    )
    candidate_counts = np.sum(problem.target_sensor_mask, axis=1)
    required = int(problem.DATA_GENERATION_MIN_COVERAGE)
    if int(np.min(candidate_counts)) < required:
        raise RuntimeError(
            f"Problem generated an invalid map for seed {seed}: "
            f"minimum coverage {int(np.min(candidate_counts))} < {required}"
        )

    dataset_dir.mkdir(parents=True, exist_ok=overwrite)
    temporary_csv = dataset_dir / ".sensor.csv.tmp"
    temporary_metadata = dataset_dir / ".metadata.json.tmp"
    temporary_svg = dataset_dir / ".sensor.svg.tmp"

    # Keep the historical CSV shape, including its index column, because
    # Problem.create_test_data reads columns 1 and 2 as x/y coordinates.
    pd.DataFrame(problem.sensor).to_csv(temporary_csv)

    metadata = {
        "dataset": dataset_name,
        "map_id": str(map_id),
        "map_seed": seed,
        "boundary": int(boundary),
        "sensors": int(sensors),
        "targets": int(targets),
        "base_station": [float(value) for value in problem.BS],
        "target_layout": "fixed_grid",
        "sensor_layout": "uniform_random_unique",
        "geometry_sha256": geometry_fingerprint(problem.sensor),
        "numpy_version": np.__version__,
        "rng": "numpy.random.RandomState/MT19937",
        "generation_attempts": int(problem.generation_attempts),
        "minimum_coverage_required": required,
        "validation": "every target has at least three sensor candidates at maximum sensing radius; not a routing proof",
        "coverage_candidates_min": int(np.min(candidate_counts)),
        "coverage_candidates_mean": float(np.mean(candidate_counts)),
    }
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_svg_preview(
        problem,
        temporary_svg,
        dataset_name,
        seed,
    )
    # Replace only after all three new files are complete, so generation
    # failures leave the previous dataset intact.
    temporary_csv.replace(dataset_dir / "sensor.csv")
    temporary_metadata.replace(dataset_dir / "metadata.json")
    temporary_svg.replace(dataset_dir / "sensor.svg")
    return dataset_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate reproducible WSN sensor maps."
    )
    parser.add_argument("--boundary", type=int, default=100)
    parser.add_argument("--sensors", type=int, default=100)
    parser.add_argument("--targets", type=int, default=100)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument(
        "--seed",
        type=int,
        default=1001,
        help="First map seed; subsequent maps use seed + index.",
    )
    parser.add_argument("--prefix", default="MAP")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--manifest", type=Path, help="Also write a --maps JSON manifest.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--validation-count", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=40001)
    parser.add_argument("--overwrite", action="store_true",
                        help="Atomically replace generated files in matching dataset folders.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.boundary <= 0 or args.sensors <= 0 or args.targets <= 0:
        raise ValueError("boundary, sensors, and targets must be positive")
    if args.count <= 0 or args.validation_count < 0:
        raise ValueError("count must be positive and validation-count non-negative")
    if args.manifest is not None and args.manifest.exists() and not args.overwrite:
        raise FileExistsError(f"manifest already exists: {args.manifest}")
    if args.manifest is not None and args.output_root.resolve() != PROJECT_ROOT / "data":
        raise ValueError("manifest datasets must be saved in the project's data directory")
    requests = [
        (args.split, f"{args.prefix}{index + 1:02d}", args.seed + index)
        for index in range(args.count)
    ] + [
        ("validation", f"{args.prefix}VAL{index + 1:02d}", args.validation_seed + index)
        for index in range(args.validation_count)
    ]
    if len({seed for _, _, seed in requests}) != len(requests):
        raise ValueError("training and validation map seed ranges must be disjoint")
    # Preflight the whole batch before creating any dataset.
    for _, map_id, _ in requests:
        destination = args.output_root / (
            f"{args.boundary}{args.sensors}{args.targets}_{map_id}"
        )
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"dataset already exists: {destination}")

    generated = []
    for _, map_id, map_seed in requests:
        generated.append(
            generate_map(
                args.boundary,
                args.sensors,
                args.targets,
                map_id,
                map_seed,
                args.output_root,
                overwrite=args.overwrite,
            )
        )
    for dataset_dir in generated:
        print(dataset_dir)
    if args.manifest is not None:
        entries = {"maps": [], "validation_maps": []}
        fingerprints = set()
        for index, dataset_dir in enumerate(generated):
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            fingerprint = metadata["geometry_sha256"]
            if fingerprint in fingerprints:
                raise ValueError("duplicate coordinates in generated batch")
            fingerprints.add(fingerprint)
            key = "maps" if index < args.count else "validation_maps"
            entries[key].append({
                "file": metadata["map_id"],
                "boundary": args.boundary, "sensors": args.sensors,
                "targets": args.targets,
            })
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_text = json.dumps({
            "split": args.split, "generator": "CreateTest.py",
            "minimum_coverage_required": Problem.DATA_GENERATION_MIN_COVERAGE,
            "target_layout": "fixed_grid", **entries,
        }, indent=2) + "\n"
        temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
        temporary_manifest.write_text(manifest_text, encoding="utf-8")
        temporary_manifest.replace(args.manifest)
        print("manifest", args.manifest)


if __name__ == "__main__":
    main()
