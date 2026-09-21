"""Experiment runner package.

Split out of the original single-file ``experiment_algorithms.py``.
``experiment_algorithms`` remains the entry point and re-exports the public
names, so existing scripts and tests keep working unchanged.

Importing this package installs the project root on ``sys.path`` and makes it
the working directory, because the datasets, maps and result folders are all
addressed relative to it.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
