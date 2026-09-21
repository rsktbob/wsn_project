"""Loading and validating the map manifests an experiment runs over."""

import json
from pathlib import Path

from experiments import PROJECT_ROOT
from experiments.config import EXPERIMENT_PRESET


def parse_file_name(value):
    if value is None:
        return None
    if str(value).lower() in ("", "none", "null"):
        return None
    return value


def default_map_spec():
    """Return the historical single-map experiment configuration.

    Only reached when a caller passes no manifest at all; ``--maps`` always
    supplies one, so this is a fallback for direct library use and tests.
    """
    return {
        "name": str(EXPERIMENT_PRESET["file"]),
        "file": parse_file_name(EXPERIMENT_PRESET["file"]),
        "boundary": int(EXPERIMENT_PRESET["boundary"]),
        "sensors": int(EXPERIMENT_PRESET["sensors"]),
        "targets": int(EXPERIMENT_PRESET["targets"]),
        "full_energy": float(EXPERIMENT_PRESET["full_energy"]),
        "map_seed": None,
    }


def load_map_specs(json_path, section="maps"):
    """Load and validate map definitions from a JSON manifest."""
    if json_path is None:
        return [default_map_spec()]
    manifest_path = Path(json_path).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    with manifest_path.open("r", encoding="utf-8") as json_file:
        document = json.load(json_file)
    entries = document.get(section) if isinstance(document, dict) else document
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"maps JSON must contain a non-empty '{section}' list")

    specs = []
    names = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"name": entry, "file": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"maps[{index}] must be a string or object")
        file_name = parse_file_name(entry.get("file", entry.get("id")))
        if file_name is None:
            raise ValueError(f"maps[{index}] requires 'file' or 'id'")
        name = str(entry.get("name", file_name)).strip()
        if not name or Path(name).name != name:
            raise ValueError(f"maps[{index}].name must be one folder-safe name")
        if name in names:
            raise ValueError(f"duplicate map name: {name}")
        names.add(name)
        spec = {
            "name": name,
            "file": file_name,
            "boundary": int(entry.get("boundary", EXPERIMENT_PRESET["boundary"])),
            "sensors": int(entry.get("sensors", EXPERIMENT_PRESET["sensors"])),
            "targets": int(entry.get("targets", EXPERIMENT_PRESET["targets"])),
            "full_energy": float(
                entry.get("full_energy", EXPERIMENT_PRESET["full_energy"])
            ),
            "map_seed": entry.get("map_seed"),
            "geometry_sha256": entry.get("geometry_sha256"),
        }
        if min(spec["boundary"], spec["sensors"], spec["targets"]) <= 0:
            raise ValueError(f"maps[{index}] dimensions must be positive")
        dataset_dir = PROJECT_ROOT / "data" / (
            f'{spec["boundary"]}{spec["sensors"]}{spec["targets"]}_{file_name}'
        )
        if not (dataset_dir / "sensor.csv").is_file():
            raise FileNotFoundError(f"map dataset does not exist: {dataset_dir}")
        metadata_path = dataset_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("map_seed", "geometry_sha256"):
                if spec.get(key) is None:
                    spec[key] = metadata.get(key)
        specs.append(spec)
    return specs


def current_map_spec(args):
    return getattr(args, "_current_map", default_map_spec())
