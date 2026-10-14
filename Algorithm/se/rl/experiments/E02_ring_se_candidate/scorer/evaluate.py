"""E02 / 監督式打分網路的 lifetime 評估：凍結的打分網路對 4 個候選打分，選最高的。

用法（在 2027WSN 根目錄）：
    E02_SCORER=.../scorer_25d_w64.pt python Algorithm/se/rl/experiments/E02_ring_se_candidate/scorer/evaluate.py \
        --algorithm e02_scorer --runs 5 --seed 7 --evaluate 1000 --maps maps/maps_100100100.json ...
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E02 = HERE.parent
ROOT = E02.parents[4]
for p in (str(HERE), str(E02), str(ROOT)):
    sys.path.insert(0, p)
os.chdir(ROOT)

import torch  # noqa: E402

import experiments.registry as registry  # noqa: E402

registry.ALGORITHM_IMPORTS["e02_scorer"] = ("evaluate", "Ring_SE_E02_Scorer")
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

ALGORITHM_PRESETS["e02_scorer"] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
ALGORITHM_PRESETS["e02_scorer"]["label"] = "Ring_SE_E02_Scorer"

from capacity import scorer  # noqa: E402
from env import Ring_SE_E02  # noqa: E402


def load_scorer_policy(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cols = ckpt["cols"]
    net = scorer(len(cols), ckpt["width"])
    net.load_state_dict(ckpt["state"])
    net.eval()

    def policy(X):
        with torch.no_grad():
            return net(torch.as_tensor(X[:, :, cols])).squeeze(-1).argmax(dim=1).numpy()
    return policy


class Ring_SE_E02_Scorer(Ring_SE_E02):
    def __init__(self, problem, **kwargs):
        torch.set_num_threads(1)
        super().__init__(problem, policy=load_scorer_policy(os.environ["E02_SCORER"]), **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
