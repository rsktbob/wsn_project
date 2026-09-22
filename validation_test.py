#!/usr/bin/env python3
"""重構安全網：跑完 tests/ 底下所有測試，並比對出「新壞掉的」。

tests/ 的每支檔案都是獨立的 `__main__` 腳本：正常結束代表通過，
assert 失敗或例外代表不通過。這支 runner 沿用該慣例，用子行程逐檔
執行（順帶隔離 numba 快取與全域狀態），加上逾時保護與平行度控制。

重構前先存一份基準：

    python validation_test.py --save-baseline tests/baseline.json

重構後比對，只看「新壞掉的」：

    python validation_test.py --compare tests/baseline.json

注意要用 wsnenv 的解譯器跑（base env 沒有 numpy）：

    ~/miniconda3/envs/wsnenv/bin/python validation_test.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TESTS_DIR = PROJECT_ROOT / "tests"

# benchmark_* 是效能量測腳本，不是通過／失敗的測試，預設不跑。
DEFAULT_EXCLUDE_PREFIXES = ("benchmark_",)

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

# environment.yml 定義的 conda env；base env 沒有 numpy，用它跑會 46 支全紅。
WSNENV_PYTHON = Path.home() / "miniconda3" / "envs" / "wsnenv" / "bin" / "python"


def resolve_python(explicit):
    """挑出能 import numpy 的解譯器，挑不到就講清楚該怎麼跑。"""
    candidates = [explicit] if explicit else [sys.executable, str(WSNENV_PYTHON)]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", "import numpy"], capture_output=True
        )
        if probe.returncode == 0:
            return candidate
    raise SystemExit(
        "找不到有 numpy 的 Python。請用 wsnenv 跑：\n"
        f"    {WSNENV_PYTHON} validation_test.py"
    )


def collect(patterns, include_benchmarks):
    paths = sorted(TESTS_DIR.glob("*.py"))
    if not include_benchmarks:
        paths = [
            path
            for path in paths
            if not path.name.startswith(DEFAULT_EXCLUDE_PREFIXES)
        ]
    if patterns:
        paths = [
            path
            for path in paths
            if any(pattern in path.name for pattern in patterns)
        ]
    return paths


def run_one(python, path, timeout):
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [python, str(path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": path.name,
            "status": TIMEOUT,
            "returncode": None,
            "seconds": round(time.perf_counter() - started, 1),
            "tail": f"逾時 {timeout}s",
        }
    tail = (completed.stderr or completed.stdout).strip().splitlines()
    return {
        "name": path.name,
        "status": PASS if completed.returncode == 0 else FAIL,
        "returncode": completed.returncode,
        "seconds": round(time.perf_counter() - started, 1),
        "tail": tail[-1] if tail else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patterns",
        nargs="*",
        help="只跑檔名含有這些字串的測試，例如 sa_sets gomea",
    )
    parser.add_argument("--jobs", "-j", type=int, default=4, help="平行行程數")
    parser.add_argument(
        "--timeout", type=int, default=600, help="單一測試逾時秒數"
    )
    parser.add_argument(
        "--include-benchmarks",
        action="store_true",
        help="連 benchmark_* 一起跑",
    )
    parser.add_argument("--save-baseline", metavar="PATH", help="把結果寫成基準快照")
    parser.add_argument("--compare", metavar="PATH", help="與基準快照比對")
    parser.add_argument("--python", help="指定解譯器，預設自動找有 numpy 的那個")
    parser.add_argument("--verbose", "-v", action="store_true", help="印出失敗訊息")
    args = parser.parse_args()

    paths = collect(args.patterns, args.include_benchmarks)
    if not paths:
        print("沒有符合的測試")
        return 1

    python = resolve_python(args.python)
    print(f"解譯器 {python}")
    print(f"跑 {len(paths)} 支測試，平行度 {args.jobs}，逾時 {args.timeout}s\n")
    results = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, python, path, args.timeout): path
            for path in paths
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            mark = {PASS: "ok  ", FAIL: "FAIL", TIMEOUT: "TIME"}[result["status"]]
            print(f"  {mark}  {result['name']:<45} {result['seconds']:>6.1f}s")

    results.sort(key=lambda result: result["name"])
    elapsed = time.perf_counter() - started
    passed = [r for r in results if r["status"] == PASS]
    broken = [r for r in results if r["status"] != PASS]

    print(f"\n{len(passed)}/{len(results)} 通過，共 {elapsed:.0f}s")
    if broken:
        print(f"\n不通過 {len(broken)} 支：")
        for result in broken:
            print(f"  {result['status']:<8} {result['name']}")
            if args.verbose and result["tail"]:
                print(f"           {result['tail']}")

    if args.save_baseline:
        snapshot = {result["name"]: result["status"] for result in results}
        Path(args.save_baseline).write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
        )
        print(f"\n基準已寫入 {args.save_baseline}")

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        current = {result["name"]: result["status"] for result in results}
        regressions = sorted(
            name
            for name, status in current.items()
            if status != PASS and baseline.get(name) == PASS
        )
        fixed = sorted(
            name
            for name, status in current.items()
            if status == PASS and baseline.get(name, PASS) != PASS
        )
        added = sorted(set(current) - set(baseline))
        if fixed:
            print(f"\n修好了 {len(fixed)} 支：" + ", ".join(fixed))
        if added:
            print(f"\n基準中沒有的新測試：" + ", ".join(added))
        if regressions:
            print(f"\n新壞掉 {len(regressions)} 支：")
            for name in regressions:
                print(f"  {name}")
            return 1
        print("\n沒有新的退步。")
        return 0

    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
