# GPU-GOMEA 使用說明

`GPU_GOMEA` 保留目前 GI-GOMEA 的交換、接受、解碼、精確路由與 fitness
流程，只將每代最耗時的 normalized mutual information（NMI）矩陣交給
CUDA。這個範圍記錄為 `linkage_nmi`，不宣稱是已經完成 graph-colored、
gray-box partial evaluation 的全文獻版 Parallel GOMEA。

## 安裝

先確認 NVIDIA driver 可使用：

```powershell
nvidia-smi
```

依電腦上的 CUDA major version 安裝其中一個 CuPy wheel，不可同時安裝：

```powershell
# CUDA 12.x
python -m pip install cupy-cuda12x

# CUDA 13.x
python -m pip install cupy-cuda13x
```

本實作使用 CuPy `RawKernel`，不要求另外以 `nvcc` 編譯專案。可以先執行：

```powershell
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
```

## 執行

```powershell
python experiment_algorithms.py --algorithm gpu_gomea --mode lifetime --evaluate 10000 --runs 5
```

GPU 設定來自 `experiments/presets.py` 的 `gpu_gomea` preset（`cuda_device`、
`require_cuda`），不再有對應的命令列參數。預設 `require_cuda: False`，
所以沒有 CUDA、CuPy 或 kernel 執行失敗時會自動退回原本的 NumPy NMI。若要
求一定使用 GPU、CUDA 初始化失敗就直接停止（避免實驗意外在 CPU 上完成），
把該 preset 的 `require_cuda` 改成 `True`；要換一張 GPU 則改 `cuda_device`。

程式結束時會顯示：

```text
device= cuda:0 (GPU name) gpu_nmi_calls= ...
```

也可以直接檢查：

```python
assert algorithm.gpu_accelerated
assert algorithm.gpu_scope == "linkage_nmi"
print(algorithm.device_detail)
```

## 驗證

以下測試會比較 CUDA 與原始 CPU NMI 的數值；在沒有 GPU 的電腦上則驗證
fallback 與 `require_cuda=True` 的錯誤處理：

```powershell
python -B tests/smoke_gpu_gomea.py
python -B tests/smoke_gi_gomea.py
python -B tests/smoke_experiment_algorithms.py
python -B tests/regression_algorithm_contract.py
```

CUDA 使用 `float64` 計算 entropy，容許誤差設為 `1e-12`。若 CPU 與 GPU
剛好遇到 linkage tree 的數值平手，浮點運算順序仍可能使搜尋軌跡不同，
因此實驗應固定種子並分別報告 CPU/GPU 裝置與執行時間。
