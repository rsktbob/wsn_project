"""E03 lifetime 評估：e03_scorer（凍結的 DeepSets 打分網路選 4 個候選中分數最高的）、e03_bottleneck（瓶頸規則）。

用法（在 2027WSN 根目錄）：
    E03_SCORER=.../deepsets_w64.pt python Algorithm/se/rl/experiments/E03_ring_se_decoded/deepsets/evaluate.py \
        --algorithm e03_scorer --runs 5 --seed 7 --evaluate 1000 --maps maps/maps_100100100.json ...
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E03 = HERE.parent
ROOT = E03.parents[4]
for p in (str(HERE), str(E03), str(ROOT)):
    sys.path.insert(0, p)
os.chdir(ROOT)

import torch  # noqa: E402

import experiments.registry as registry  # noqa: E402

registry.ALGORITHM_IMPORTS["e03_scorer"] = ("evaluate", "Ring_SE_E03_Scorer")
registry.ALGORITHM_IMPORTS["e03_bottleneck"] = ("evaluate", "Ring_SE_E03_Bottleneck")
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

for name, label in (("e03_scorer", "Ring_SE_E03_Scorer"), ("e03_bottleneck", "Ring_SE_E03_Bottleneck")):
    ALGORITHM_PRESETS[name] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
    ALGORITHM_PRESETS[name]["label"] = label

from env import Ring_SE_E03  # noqa: E402
from model import DeepSetsScorer  # noqa: E402


def load_scorer_policy(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = DeepSetsScorer(width=ckpt["width"])
    net.load_state_dict(ckpt["state"])
    net.eval()

    def policy(batch):
        with torch.no_grad():
            scores = net(torch.as_tensor(batch["tokens"]), torch.as_tensor(batch["mask"]),
                         torch.as_tensor(batch["extra"]))
        return scores.argmax(dim=1).numpy()
    return policy


class Ring_SE_E03_Scorer(Ring_SE_E03):
    def __init__(self, problem, **kwargs):
        torch.set_num_threads(1)
        super().__init__(problem, policy=load_scorer_policy(os.environ["E03_SCORER"]), **kwargs)


class Ring_SE_E03_Bottleneck(Ring_SE_E03):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="bottleneck", **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
