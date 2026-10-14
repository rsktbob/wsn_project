"""以 CUDA 平行計算 GI-GOMEA 的 normalized mutual information。

CuPy 只在要求 CUDA 時才延遲匯入，因此原本的 CPU 演算法不需要安裝
CUDA 或 CuPy。核心計算使用 ``RawKernel``，每個 CUDA thread 負責一組
基因配對，避免在 Python 中逐對呼叫 ``numpy.unique``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_NMI_KERNEL = r"""
extern "C" __global__
void nmi_matrix(
    const long long* codes,
    const int sample_count,
    const int gene_count,
    double* similarity
) {
    const long long output_id =
        (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long output_count =
        (long long)gene_count * (long long)gene_count;
    if (output_id >= output_count) {
        return;
    }

    const int first = (int)(output_id / gene_count);
    const int second = (int)(output_id % gene_count);
    if (first == second) {
        similarity[output_id] = 1.0;
        return;
    }
    // 只計算上三角；同一個 thread 同步寫入對稱位置，工作量減半。
    if (first > second) {
        return;
    }

    const double inverse_sample_count = 1.0 / (double)sample_count;
    double first_entropy = 0.0;
    double second_entropy = 0.0;
    double joint_entropy = 0.0;

    // 對每筆樣本計算它所屬類別的出現次數。若某類別共有 c 筆，
    // -log(c / N) / N 會被累加 c 次，正好等於 -p * log(p)。
    for (int sample = 0; sample < sample_count; ++sample) {
        const long long first_value =
            codes[(long long)sample * gene_count + first];
        const long long second_value =
            codes[(long long)sample * gene_count + second];
        int first_count = 0;
        int second_count = 0;
        int joint_count = 0;

        for (int other = 0; other < sample_count; ++other) {
            const long long other_first =
                codes[(long long)other * gene_count + first];
            const long long other_second =
                codes[(long long)other * gene_count + second];
            const bool same_first = other_first == first_value;
            const bool same_second = other_second == second_value;
            first_count += (int)same_first;
            second_count += (int)same_second;
            joint_count += (int)(same_first && same_second);
        }

        first_entropy -= log(first_count * inverse_sample_count)
            * inverse_sample_count;
        second_entropy -= log(second_count * inverse_sample_count)
            * inverse_sample_count;
        joint_entropy -= log(joint_count * inverse_sample_count)
            * inverse_sample_count;
    }

    if (joint_entropy <= 1.0e-15) {
        similarity[output_id] = 0.0;
        similarity[(long long)second * gene_count + first] = 0.0;
        return;
    }

    const double mutual_information =
        first_entropy + second_entropy - joint_entropy;
    const double nmi = fmax(0.0, mutual_information / joint_entropy);
    similarity[output_id] = nmi;
    similarity[(long long)second * gene_count + first] = nmi;
}
"""


class CudaUnavailableError(RuntimeError):
    """表示要求的 CUDA/CuPy 執行環境不存在或無法初始化。"""


@dataclass(frozen=True)
class CudaDeviceInfo:
    """保存實驗輸出需要的 CUDA 裝置資訊。"""

    device_id: int
    name: str


class CudaNMI:
    """管理 CuPy CUDA context，並執行 NMI RawKernel。"""

    THREADS_PER_BLOCK = 256

    def __init__(self, device_id=0):
        """延遲載入 CuPy 並選擇指定的 NVIDIA GPU。"""
        try:
            import cupy as cp
        except (ImportError, ModuleNotFoundError) as error:
            raise CudaUnavailableError(
                "找不到 CuPy；CUDA 12 請安裝 cupy-cuda12x，"
                "CUDA 13 請安裝 cupy-cuda13x。"
            ) from error

        self.cp = cp
        self.device_id = int(device_id)
        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
            if device_count <= 0:
                raise CudaUnavailableError("CuPy 找不到可用的 NVIDIA GPU。")
            if not 0 <= self.device_id < device_count:
                raise CudaUnavailableError(
                    f"CUDA device {self.device_id} 不存在；"
                    f"目前只有 {device_count} 個裝置。"
                )
            self.device = cp.cuda.Device(self.device_id)
            self.device.use()
            properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        except CudaUnavailableError:
            raise
        except Exception as error:
            raise CudaUnavailableError(
                f"CUDA device {self.device_id} 初始化失敗：{error}"
            ) from error

        raw_name = properties.get("name", properties.get(b"name", "CUDA GPU"))
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", errors="replace")
        self.info = CudaDeviceInfo(self.device_id, str(raw_name))
        self.kernel = cp.RawKernel(_NMI_KERNEL, "nmi_matrix")

    def similarity(self, codes):
        """回傳與 CPU 定義相同的 NMI 矩陣，輸出仍為 NumPy array。"""
        host_codes = np.ascontiguousarray(codes, dtype=np.int64)
        if host_codes.ndim != 2:
            raise ValueError("codes 必須是二維的 population-by-gene 矩陣。")
        sample_count, gene_count = host_codes.shape
        if sample_count <= 0 or gene_count <= 0:
            raise ValueError("codes 不可包含零個樣本或零個基因。")

        cp = self.cp
        with self.device:
            device_codes = cp.asarray(host_codes)
            device_similarity = cp.empty(
                (gene_count, gene_count),
                dtype=cp.float64,
            )
            output_count = gene_count * gene_count
            block_count = (
                output_count + self.THREADS_PER_BLOCK - 1
            ) // self.THREADS_PER_BLOCK
            self.kernel(
                (block_count,),
                (self.THREADS_PER_BLOCK,),
                (
                    device_codes,
                    np.int32(sample_count),
                    np.int32(gene_count),
                    device_similarity,
                ),
            )
            # asnumpy 同時形成唯一一次 GPU -> CPU 傳輸並等待 kernel 完成。
            return cp.asnumpy(device_similarity)
