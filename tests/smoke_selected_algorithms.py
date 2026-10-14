import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CASES = [
    ("SA-SETS", "tests/smoke_sa_sets.py"),
    ("Ring-SE", "tests/smoke_ring_se.py"),
    ("GA", "tests/smoke_ga.py"),
    ("GA variants", "tests/test_ga_variants.py"),
    ("PSO", "tests/test_pso.py"),
]


def main():
    for name, script in CASES:
        print("==", name, "==")
        subprocess.run(
            [sys.executable, "-B", str(PROJECT_ROOT / script)],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    print("smoke_selected_algorithms_ok")


if __name__ == "__main__":
    main()
