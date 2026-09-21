"""固定種子下的決定性檢查。

原本是 radius 重構期間的一次性工具（先 capture 基準、再 compare）。
那次重構已結束，因此無參數執行時改為自我檢查：同一份程式碼連續跑
兩次 capture()，逐欄位比對必須完全相同。任何破壞決定性的改動
（未播種的亂數、對雜湊順序的依賴、平行化引入的競爭）都會在這裡現形。

仍保留舊用法以便跨版本比對：
    python tests/check_radius_refactor.py capture <path>
    python tests/check_radius_refactor.py compare <path>
"""
import sys
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_algorithms import parse_args, build_problem, build_algorithm
from State.SensorEncoding import SensorEncoding


def capture():
    records = []
    for routing in ('v1',):
        args = parse_args(['--algorithm', 'sa_sets', '--routing-service', routing])
        problem = build_problem(args, 7)
        rng = np.random.default_rng(81)
        for scale in (1.0, 0.000001):
            problem.energy[:] = problem.initial_energy * scale
            problem.build_capacities()
            problem.prepare_coding_cache()
            for version in ('v2', 'v3'):
                for _ in range(20):
                    coding = SensorEncoding.random(problem.SENSOR_NUMBER, problem.radius_option_counts, rng=rng)
                    state = getattr(coding, 'decode' + version)(problem)
                    values = problem.evaluate_state(state)
                    records.append((coding.code, state.levels, state.next_hops, state.tx_load,
                                    state.remaining_capacity, state.sensing_radii, state.paths, values))
    for seed in (7, 11):
        args = parse_args(['--algorithm', 'sa_sets'])
        problem = build_problem(args, seed)
        algorithm = build_algorithm('sa_sets', problem, args, seed)
        result = algorithm.run(problem, budget=10000)
        state = result.best_state
        records.append((state.levels, state.next_hops, state.tx_load, state.remaining_capacity,
                        state.sensing_radii, state.paths, result.best_fitness,
                        result.evaluations, result.history, algorithm.final_coding.code))
    return records


def equal(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        # 無參數：自我檢查決定性，讓這支程式能在測試套件裡直接跑。
        first = capture()
        second = capture()
        equal(first, second)
        print('check_radius_refactor_ok', len(first))
    else:
        path = Path(sys.argv[2])
        actual = capture()
        if sys.argv[1] == 'capture':
            path.write_bytes(pickle.dumps(actual))
            print('baseline captured', len(actual))
        else:
            equal(pickle.loads(path.read_bytes()), actual)
            print('EXACT MATCH', len(actual))
