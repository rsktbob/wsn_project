"""GPU-GOMEA 的 CUDA/CPU fallback 與 NMI 數值煙霧測試。"""

import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.gomea.GI_GOMEA import GI_GOMEA
from Algorithm.gomea.GPU_GOMEA import GPU_GOMEA
from Algorithm.gomea.cuda_nmi import CudaUnavailableError
from Problem.Problem import Problem


def main():
    np.random.seed(19)
    random.seed(19)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    cpu = GI_GOMEA(
        problem,
        population_size=8,
        max_linkage_size=8,
        max_linkage_sets=24,
        seed=19,
    )
    gpu = GPU_GOMEA(
        problem,
        population_size=8,
        max_linkage_size=8,
        max_linkage_sets=24,
        seed=19,
        require_cuda=False,
    )

    # 使用不同 domain 與重複類別，涵蓋零熵、完全關聯及部分關聯。
    codes = np.asarray(
        [
            [0, 0, 4, 7, 1],
            [0, 1, 4, 8, 1],
            [1, 0, 3, 7, 1],
            [1, 1, 3, 9, 1],
            [2, 0, 2, 8, 1],
            [2, 1, 2, 9, 1],
        ],
        dtype=np.int64,
    )
    expected = cpu._normalized_mutual_information(codes)
    actual = gpu._normalized_mutual_information(codes)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=1.0e-15)
    np.testing.assert_array_equal(np.diag(actual), np.ones(codes.shape[1]))

    if gpu.gpu_accelerated:
        assert gpu.device_name == "cuda"
        assert gpu.gpu_scope == "linkage_nmi"
        assert gpu.nmi_gpu_calls == 1
    else:
        assert gpu.device_name == "cpu"
        assert gpu.cuda_error
        assert gpu.nmi_cpu_calls == 1
        try:
            GPU_GOMEA(problem, seed=19, require_cuda=True)
        except CudaUnavailableError:
            pass
        else:
            raise AssertionError("require_cuda=True 應在無 CUDA 時立即失敗。")

    result = gpu.run(problem, budget=40)
    assert result.best_state is not None
    assert not hasattr(result, "best_coding")
    assert 38 <= gpu.evatime <= 40
    assert gpu.gene_invariance_error() == 0
    assert gpu.acceleration_scope == "linkage_nmi"
    print(
        "smoke_gpu_gomea_ok",
        "device=",
        gpu.device_detail,
        "gpu_nmi_calls=",
        gpu.nmi_gpu_calls,
    )


if __name__ == "__main__":
    main()
