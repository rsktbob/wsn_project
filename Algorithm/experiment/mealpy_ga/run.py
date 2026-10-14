"""Direct entry to MEALPY GA on WSN, using the shared experiment runner."""
from pathlib import Path
import os
import sys
# Small WSN matrices benefit from one BLAS thread; respect explicit overrides.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    from experiment_algorithms import main as run_experiment
    # All shared CLI options remain available. This entry always uses MEALPY GA.
    if any(arg == '--algorithm' or arg.startswith('--algorithm=') for arg in sys.argv[1:]):
        raise SystemExit('This entry runs mealpy_ga; use experiment_algorithms.py to compare algorithms.')
    sys.argv.extend(['--algorithm', 'mealpy_ga'])
    defaults = {'--mode': 'single', '--runs': '1', '--evaluate': '1000',
                '--result-dir': str(ROOT / 'Algorithm/experiment/results/mealpy_ga')}
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
            sys.argv.extend([flag, value])
    run_experiment()


if __name__ == '__main__':
    main()
