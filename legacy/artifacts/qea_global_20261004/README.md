# 全局 QEA 初步比較（2026-10-04）

已實作集中式、全網路感測等級與路由優先序的 QEA。這是傳統 CPU 上的量子啟發式搜尋。

MAP01–MAP03，每張五個種子（7–11），各方法每次 **3,006 次實際 fitness 評估**，
共 60 次搜尋。使用 fitness v2、discrete sensing、共用 SensorEncoding v3 解碼器。

| 方法 | 平均最佳 fitness（越高越好） | 平均秒數 | 可行解 |
|---|---:|---:|---:|
| qea | 0.998253484 | 0.747 | 15/15 |
| qea_classical | 0.997907698 | 0.762 | 15/15 |
| qea_random | 0.998057885 | 0.708 | 15/15 |
| ga | 0.998250502 | 0.416 | 15/15 |

QEA 對 qea_classical：14 勝、1 負；對 qea_random：14 勝、1 負；對 GA：5 勝、10 負。
QEA 與 GA 的平均 fitness 差只有 +0.000002981，QEA 平均耗時約為 GA 的 1.79 倍。
**這批結果沒有建立 QEA 優於 GA 的證據。** QEA 較兩個機率對照好，但只適用於
這次參數設定；尚未做學習率調參、統計推論、更多地圖或完整壽命比較。

三個 QEA 版本使用相同均勻初始分布、編碼、族群數與最佳解更新遮罩；GA 沿用
既有 coverage-seeded 初始化。沒有修改 GA，選擇其整代評估可剛好用完的預算。
所有方法均先使用共用核心暖機；計時包含搜尋與逐次記錄，不包含 Problem 建立。

- `results.csv`：每次執行的原始指標、耗能、違規數與時間。
- `summary.json`：參數、地圖、種子、平均值與成對勝負。
- `histories.npz`：每次評估的歷史最佳 fitness。
- `convergence.png`：各方法平均收斂曲線。
- `lifetime_smoke/`：接入原本壽命流程的兩時槽驗證，非壽命優劣實驗。

驗證通過：smoke_qea.py、smoke_algorithm_catalog.py、smoke_experiment_algorithms.py。
QEA 測試涵蓋三種 sensing mode、合法性與初始均勻分布、旋轉方向、振幅正規化、
小於族群與非整代預算、相同 seed 重現性、三種對照首批一致、迭代上限與重複執行。

重跑：

```bash
/home/g114056067/miniconda3/envs/wsnenv/bin/python /home/g114056067/wsn_project/codes/2027WSN/experiments/benchmark_qea.py --map-count 3 --runs 5 --budget 3006 --output /tmp/qea_recheck
```

方法與一般 CLI 詳見 `codes/2027WSN/docs/global_qea.md`。
