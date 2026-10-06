"""Run two short training maps and frozen validation through the real pipeline."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from train_rl_sets import main
from Algorithm.se.RL_SETS import RL_SETS
from Algorithm.se.RL_SETSv5 import RL_SETSv5


def run_smoke(extra_options=()):
    with tempfile.TemporaryDirectory() as temp:
        model = Path(temp) / "smoke.npz"
        manifest = Path(temp) / "maps.json"
        manifest.write_text(json.dumps({
            "maps": ["RLTRAIN01", "RLTRAIN02"],
            "validation_maps": ["RLTRAINVAL01"],
        }), encoding="utf-8")
        options = [
            "--rl-model", str(model), "--maps", str(manifest),
            "--episodes", "2", "--evaluate", "40", "--max-rounds", "1",
            "--segment-slot-limit", "1", "--max-lifetime", "1",
            "--n", "4", "--w", "1", "--replay-warmup", "4",
            "--batch-size", "4", "--target-update-interval", "2",
            "--validation-every", "2", "--validation-seeds", "7007",
        ]
        options.extend(extra_options)
        summaries = main(options)
        assert len(summaries) == 2
        assert len({row["map"] for row in summaries}) == 2
        assert [row["seed"] for row in summaries] == [7, 8]
        assert all(row["actual_evaluations"] == 40 for row in summaries)
        report = json.loads(model.with_suffix(".training.json").read_text())
        assert len(report["validations"]) == 1
        validation = report["validations"][0]["rows"][0]
        assert validation["map"] == "RLTRAINVAL01"
        assert validation["policy"]["epsilon"] == 0
        assert validation["policy"]["replay_size"] == 0
        assert model.with_name("smoke.best.npz").is_file()
        with np.load(model, allow_pickle=False) as checkpoint:
            assert int(checkpoint["environment_steps"]) == 32
            assert int(checkpoint["training_steps"]) == 8
            # checkpoint 已統一成單一 metadata_json；舊的 v2_metadata_json 分支
            # 對應的欄位不再寫入。
            metadata = json.loads(str(checkpoint["metadata_json"]))
            assert metadata["completed_episodes"] == 2
            # 對照演算法自己宣告的常數，避免又寫死一個會過期的字串。
            if "rl_sets" in extra_options:
                expected_reward = RL_SETS.REWARD_VERSION
            else:
                expected_reward = RL_SETSv5.REWARD_VERSION
            assert metadata["reward_version"] == expected_reward
        for row in summaries:
            assert sum(action["decisions"] for action in row["episode_actions"].values()) == 16
            assert "remaining_80_percent" in row["reward_by_progress"]
        # Resume skips already completed maps and preserves the checkpoint.
        resumed = main(options + ["--resume"])
        assert len(resumed) == 2
        # A stopped validation can be recovered without replaying training maps.
        report["validations"] = []
        report["best_validation_mean_lifetime"] = None
        model.with_suffix(".training.json").write_text(json.dumps(report), encoding="utf-8")
        main(options + ["--resume"])
        recovered = json.loads(model.with_suffix(".training.json").read_text())
        assert len(recovered["validations"]) == 1
        extended = main(options + ["--resume", "--episodes", "3"])
        assert len(extended) == 3 and extended[-1]["seed"] == 9
        assert extended[-1]["policy"]["training_steps"] > summaries[-1]["policy"]["training_steps"]
    print("smoke_train_rl_sets_ok")


if __name__ == "__main__":
    run_smoke()
    run_smoke(("--algorithm", "rl_sets"))
