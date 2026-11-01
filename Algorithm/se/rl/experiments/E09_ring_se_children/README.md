# E09：只看子代的 Transformer 選候選範圍（ChildNet）

- 狀態（10/25）：b1000 資料已收（200 張）、ChildNet（plain_delta、soft τ=1、全 0 也訓練）訓練中；lifetime 未跑。
- 動機：E07／E08 用父代的解碼狀態＋摘要，test 正值交易選到最大卡在 50%／54%，lifetime 和 E07 沒差。
  使用者 10/25 決定：交換後父代的狀態已不成立，**完全不用父代的資訊**，直接看兩個子代完整的 100 顆 sensor；
  也不給範圍位置（給完整子代就不需要知道從哪換）。
- 和 E07 共用：collect 流程（同 seed、同地圖、同每 3 段標一段）、run_lifetime、build_ring、loss_fn。

## 輸入與網路

每個候選一條序列：`[CLS1] child1 的 100 顆 [CLS2] child2 的 100 顆`（202 個 token），4 個候選共用權重。

| token（每顆 sensor，13 維） | 來源 |
| --- | --- |
| 這個子代的等級基因 % 6 ÷ 5、優先度基因 ÷ 9 | 父代原始染色體直接拼（不解碼） |
| 剩餘電量比例、到 BS 距離 | E07 `sensor` |
| energy_score（電量 ÷ 全網最大）、proximity_score | 由電量算／地圖常數 |
| 等級 1–5 各能覆蓋幾個 target ÷ T、座標 x、y ÷ 邊界 | 地圖常數（`results/map_const.npz`） |

＋ 位置嵌入（sensor 編號，兩個子代共用）＋ 子代嵌入；Transformer 3 層（d 64、4 頭、FFN 128、pre-norm）。
CLS1 ‖ CLS2 ‖ 全域 2（搜尋進度、活著 sensor 平均電量）→ 130 → 128 → 64 → 1；4 個分數 softmax。
13.3 萬參數。第一版沒有距離注意力偏置（每條序列的地圖不同，偏置要逐筆算，記憶體太大）。

驗證：重收的 RLTRAIN01 和 E07 每個欄位都相同（只多 S_code、G_code）；子代基因 32/32 和手算一致。
推論成本：一回合 64 組 × 4 候選 18.8 ms（GPU、bf16）；這一回合實際評估 128 個子代約 23.6 ms（184 µs／次）。

## 檔案

- `collect_codes.py`、`run_collect.sh`：收資料（E07 欄位＋`S_code`、`G_code`）→ `results/data/b{B}/`。
- `childdata.py`：地圖常數、`ChildData`（在 GPU 上即時拼子代 token）。
- `childnet.py`：ChildNet、`load_childnet`。
- `train_child.py`：訓練＋test（同分算對 `acc_all`、`acc_pos`、`label_ratio`）→ `results/child_b{B}_m{N}_{label}{loss}/seed0/`。

## 結果（b1000、plain_delta、soft τ=1、全 0 也訓練；test 30 張，同分算對）

| 模型 | 全部交易 | 正值交易選到最大 | 標籤比 |
| --- | --- | --- | --- |
| 隨機 | 39.1% | 25.2% | 31.4% |
| E07 SupNet | 59.29% | 50.04% | 57.31% |
| E08 RelNet | 62.43% | 53.90% | 61.20% |
| E09 第一版（不減平均，bf16，7 epoch 早停） | 45.10% | 32.63% | 37.82% |
| E09 contrast（fp32，8 epoch） | 54.30% | 43.92% | 50.80% |

- 第一版 loss 一直是 ln4：同一組交易 4 條序列只差 ~20/200 個 token，4 個 CLS 幾乎一樣（tmp/e09_diag）。
- contrast：val 正值 42.8%（ep1）→ 44.1%（ep8），train loss ≈ val loss，欠擬合；每 epoch 10–15 分鐘。
- 只看子代基因比 E07／E08 差：父代的解碼狀態（誰開、路由）提供的資訊，Transformer 從原始基因學不回來。
