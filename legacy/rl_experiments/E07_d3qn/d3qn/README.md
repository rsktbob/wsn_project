# E07 / d3qn：從零的 D3QN（每段預算 1000）

- 狀態：**已停止**（2026-10-19 約 3 張圖後停掉：使用者目前不做強化學習；程式保留，之後要用再跑 `../run_rl.sh`）
- 框架：PyTorch 2.x
- 程式：`../train_rl.py`（設定 `../config.json` 的 `rl`）；結果：`../results/d3qn_b1000/seed{seed}/`
- 網路圖：`network.svg`（`../make_svg.py` 產生）

## 設定

設計圖原本是每段預算 10000；監督式發現「訓練預算要和測試一樣」（實驗層 README 8.3）後，使用者 10/19 改成**每段預算 1000**，其他照設計圖：

- 網路：`../model.py` 的 DuelingNet（37,442 參數）：編碼同監督式；A_c ＝ u_c → 1、V ＝ 平均(u_1…u_4) → 64 → 1、Q_c ＝ V + A_c − mean(A)。
- 環境：RLTRAIN01–30 每張從滿電跑完整 lifetime，每段預算 1000；一段最佳化（約 8 回合）＝一個 episode；只跑 1 輪（約 950 個 episode），地圖順序打亂。
  環境用 `RoundRecorder(peek=False)`：只記錄，不偷看。
- 獎勵：同 E06 差異獎勵（實際子代、含突變），÷ 上一回合一般 Δ 的正值中位數、上限 5。
- 目標：γ = 0（目標就是 r，Huber）；所以沒有 target network、Double DQN、n-step。Dueling、比例式 PER α 0.4、β 0.6→1，在 GPU 上抽樣。
- Replay：FIFO 上限 15,000 回合（預算 1000 時整個訓練只約 7,600 回合，等於全留，同 E06）。
- 設定：從零、ε 1.0 → 0.05（前 50% 的預期 960 個 episode 線性下降）、warmup 5,000 組、batch 128、lr 1e-4、L2 1e-5、梯度裁剪 10。
  **每回合更新 16 次**（設計圖的 4 次是配 10000 的總量；改 1000 後改成同 E06 的 16 次，共約 12 萬次更新）。
- 不驗證；每 100 episode 存 `ep{k}.pt`，評估用 `final.pt`；log 記錄每張訓練圖的 lifetime（含 ε 探索）、每段改善率、4 個候選位置被選比例。
- 評估：預算 1000，MAP01–06 × seed 7–11，選 Q 最大 → `experiment_results/s711_e07_d3qn_b1000_eval1000`。

## 結果

（未跑完，沒有結果）
