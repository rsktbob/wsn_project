"""所有演算法共用的初始解：sensor 編碼與 priority 編碼各一個函式。

    from State.initialization import random_sensor_coding, random_priority_coding

    candidate = random_sensor_coding(problem, self.rng, self.random)
    candidate = random_priority_coding(problem, self.rng)
"""

from __future__ import annotations

from State.PriorityEncoding import PriorityEncoding
from State.SensorEncoding import SensorEncoding


def random_sensor_coding(problem, rng, random, ccs=True):
    """SensorEncoding 的初始解；預設加上 CCS。

    先全部隨機（level、priority 都均勻抽）。``ccs=True`` 時再讓每個 target
    從 ``problem.cover_candidates`` 隨機抽一組 (sensor, level)，把該 sensor 的
    level 設成剛好覆蓋得到的最小值；後面的 target 抽到同一顆 sensor 時，以後一次
    為準。路由 priority 維持隨機。``ccs=False`` 只做第一步。
    """
    coding = SensorEncoding.random(
        problem.SENSOR_NUMBER,
        problem.radius_option_counts,
        rng=rng,
    )
    if ccs:
        for target_candidates in problem.cover_candidates:
            if target_candidates:
                sensor_id, level = random.choice(target_candidates)
                coding.code[int(sensor_id) * 2] = int(level)
    return coding


def random_priority_coding(problem, rng):
    """PriorityEncoding 的初始解：每顆 sensor 的開啟、路由兩個基因都均勻抽 0–9。

    這個編碼沒有 level（由解碼器決定），所以不做 CCS。
    """
    return PriorityEncoding.random(problem.SENSOR_NUMBER, rng=rng)


__all__ = ["random_priority_coding", "random_sensor_coding"]
