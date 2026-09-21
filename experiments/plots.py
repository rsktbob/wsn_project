"""Optional per-run figures. Matplotlib/Pillow are imported only when used."""

import os
from pathlib import Path

import numpy as np

from experiments import PROJECT_ROOT


def save_fitness_history_plot(
    history,
    output_path,
    *,
    algorithm_name,
    segment_id,
    lifetime_start,
):
    """Save one optimizer call's best-so-far fitness history."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(PROJECT_ROOT / "tmp" / "matplotlib"),
    )
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    # Some optimizers (notably GA) record the best individual in each
    # generation, which can decrease after mutation.  The chart consistently
    # reports the best solution discovered up to each evaluation.
    values = np.maximum.accumulate(np.asarray(history, dtype=float))
    evaluations = np.arange(1, len(values) + 1)
    figure, (raw_axis, gap_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        constrained_layout=True,
    )
    raw_axis.step(evaluations, values, where="post", color="tab:blue")
    raw_axis.set_title(
        f"{algorithm_name}: segment {segment_id}, lifetime start {lifetime_start}"
    )
    raw_axis.set_ylabel("best fitness")
    raw_axis.grid(alpha=0.3)

    gap_axis.step(
        evaluations,
        np.maximum(1.0 - values, 1e-12),
        where="post",
        color="tab:blue",
    )
    gap_axis.set_xlabel("fitness evaluations")
    gap_axis.set_ylabel("1 - best fitness (log scale)")
    gap_axis.set_yscale("log")
    gap_axis.grid(alpha=0.3, which="both")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
