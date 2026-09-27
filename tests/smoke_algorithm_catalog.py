"""registry、presets、parameter_basis.json 三邊的演算法名稱必須一致。

改名或刪除演算法時只改其中一邊，說明文字就會錯位到別的演算法上
（曾發生在 sa_setsv3／v4 與 RL-SETS 系列重新編號時），這支測試負責攔下來。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.presets import ALGORITHM_PRESETS
from experiments.registry import ALGORITHM_IMPORTS


def main():
    basis = json.loads(
        (PROJECT_ROOT / "experiments" / "parameter_basis.json").read_text(
            encoding="utf-8"
        )
    )
    registry, presets = set(ALGORITHM_IMPORTS), set(ALGORITHM_PRESETS)
    assert registry == presets, (
        f"只在 registry：{sorted(registry - presets)}；"
        f"只在 presets：{sorted(presets - registry)}"
    )
    assert registry == set(basis), (
        f"缺少說明：{sorted(registry - set(basis))}；"
        f"多餘說明（演算法已不存在）：{sorted(set(basis) - registry)}"
    )
    labels = [preset["label"] for preset in ALGORITHM_PRESETS.values()]
    duplicated = sorted({label for label in labels if labels.count(label) > 1})
    assert not duplicated, f"label 重複：{duplicated}"
    print("smoke_algorithm_catalog_ok")


if __name__ == "__main__":
    main()
