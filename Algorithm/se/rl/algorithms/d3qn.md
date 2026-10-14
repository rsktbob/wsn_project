# D3QN（Dueling Double DQN）

DQN 的兩個改良合在一起：

## Double DQN（van Hasselt et al., 2016）

DQN 的目標 `r + γ max_a Q_target(s', a)` 用同一個網路挑動作又估值，會系統性高估。
Double DQN 用 online 網路挑、target 網路估：

```
a* = argmax_a Q_online(s', a)
y  = r + γ Q_target(s', a*)
```

## Dueling 架構（Wang et al., 2016）

把 Q 拆成狀態價值 V 和各動作的優勢 A：

```
Q(s, a) = V(s) + A(s, a) − mean_a' A(s, a')
```

減去平均是為了讓 V、A 可識別。很多動作的 Q 差不多時（只有少數狀態要緊），比直接輸出 Q 好學。

## 其他標準元件

- Experience replay：均勻抽樣的回放緩衝區，打散相關性。
- Target network：每 C 步（或 soft update τ）同步一次。
- ε-greedy 探索。
- 損失：Huber（smooth L1）。

## 實驗 README 要寫的參數

γ、學習率、batch、replay 大小／warmup、target 更新頻率（或 τ）、ε 排程、隱藏層寬度。
