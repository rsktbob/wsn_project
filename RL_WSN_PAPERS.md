# 強化式學習與感測網路論文整理

整理日期：2026-06-28

這份文件整理前面討論過，以及另外補查的強化式學習用於 Sensor Network / Wireless Sensor Network 的論文。重點放在這些論文解決什麼問題，以及它們可以怎麼參考到目前的 `2023wsn` 程式。

## 縮寫

- WSN：Wireless Sensor Network，無線感測網路。
- MWSN：Mobile Wireless Sensor Network，行動式無線感測網路。
- UWSN：Underwater Wireless Sensor Network，水下無線感測網路。
- WRSN：Wireless Rechargeable Sensor Network，可充電無線感測網路。
- RL：Reinforcement Learning，強化式學習。
- DRL：Deep Reinforcement Learning，深度強化式學習。
- MARL：Multi-Agent Reinforcement Learning，多代理人強化式學習。

## 論文整理

| 題名 | 年份 | 主要問題 | 方法 | 簡略說明與可參考方向 |
| --- | --- | --- | --- | --- |
| [Energy-Efficiency Routing algorithms in Wireless Sensor Networks: a Survey](https://arxiv.org/abs/2002.07178) | 2020 | WSN 節能路由綜述 | Survey | 不是 RL 論文，但整理 WSN routing 為什麼要重視 energy、network lifetime、battery replacement。可當作我們設計 routing reward 的背景依據。 |
| [Energy Efficiency in Reinforcement Learning for Wireless Sensor Networks](https://arxiv.org/abs/1812.02538) | 2018 | 感測系統在學習時的能耗控制 | SARSA | 用 SARSA 讓系統在保持定位能力時降低能源成本。可參考它把「學習效果」和「能源消耗」同時放進 reward 的想法。 |
| [Anypath Routing Protocol Design via Q-Learning for Underwater Sensor Networks](https://arxiv.org/abs/2002.09623) | 2020 | UWSN 路由、延遲、壽命 | Q-learning | 用節點剩餘能量與深度資訊設計 Q-learning forwarding policy。和我們目前的 `RQLearning` 很接近，可參考 reward：能量、距離/深度進展、延遲。 |
| [Energy-Efficient Routing Algorithm for Wireless Sensor Networks: A Multi-Agent Reinforcement Learning Approach](https://arxiv.org/abs/2508.14679) | 2025 | WSN 多跳路由與 cluster-head 選擇 | MARL / Q-learning | 每個 sensor 當 agent，依剩餘能量、到 sink 距離、hop count、hotspot proximity 選 routing action。很適合作為我們 next-hop Q-learning 的延伸方向。 |
| [Game-Theoretic and Reinforcement Learning-Based Cluster Head Selection for Energy-Efficient Wireless Sensor Network](https://arxiv.org/abs/2508.12707) | 2025 | cluster head 選擇與節點壽命 | Game theory + RL | 把 cluster head selection 做成多代理人決策，目標是避免特定節點過早耗盡。若未來要做 cluster-based WSN，可以參考。 |
| [Adaptive Vision-Based Coverage Optimization in Mobile Wireless Sensor Networks: A Multi-Agent Deep Reinforcement Learning Approach](https://arxiv.org/abs/2508.14676) | 2025 | MWSN 覆蓋最佳化與移動控制 | MARL / DRL | 用多代理人 DRL 讓 mobile sensors 自主移動以提升 coverage、降低 redundancy。可參考到 `MobilityService` 或未來 mobile sensor relocation。 |
| [Deep Reinforcement Learning-Based Topology Optimization for Self-Organized Wireless Sensor Networks](https://arxiv.org/abs/1910.14199) | 2019 | WSN topology control | DRL + Monte Carlo tree search | 用深度網路引導搜尋，處理 self-organized WSN 的 topology optimization。適合參考成「解碼器 + search」混合設計，但對目前專案來說實作成本較高。 |
| [Survey on Multi-Agent Q-Learning frameworks for resource management in wireless sensor network](https://arxiv.org/abs/2105.02371) | 2021 | WSN resource management 綜述 | Multi-agent Q-learning survey | 介紹 multi-agent Q-learning、game theory framework、resource allocation 與 task scheduling。適合當作 MARL 設計入口。 |
| [Two Timescale Convergent Q-learning for Sleep-Scheduling in Wireless Sensor Networks](https://arxiv.org/abs/1312.7292) | 2013 | sensor sleep scheduling 與 tracking accuracy | Two-timescale Q-learning | 把 sleep time scheduling 視為 POMDP，在節能與 tracking error 間取平衡。可參考到我們的 sensor scheduling 或 coverage/sleep-wake 策略。 |
| [Reinforcement Learning based Transmission Range Control in Software-Defined Wireless Sensor Networks with Moving Sensor](https://arxiv.org/abs/2005.08215) | 2020 | 傳輸距離/功率控制 | epsilon-greedy RL | 針對 SD-WSN 中 moving sensor 的傳輸範圍控制，目標是降低能源消耗並維持 throughput。可參考到 routing level 或 transmission power level 的選擇。 |
| [Reinforcement Learning-Enabled Reliable Wireless Sensor Networks in Dynamic Underground Environments](https://arxiv.org/abs/1908.05804) | 2019 | 地下 WSN 在動態土壤環境下的可靠傳輸 | RL + Hidden Markov Model | 用 HMM 建模環境變化，再用 RL 決定 transmission policy，降低 packet loss。可參考「環境動態變化」時如何讓 policy 適應。 |
| [Relational Deep Reinforcement Learning for Routing in Wireless Networks](https://arxiv.org/abs/2012.15700) | 2020 | 一般 wireless network routing | Relational DRL | 不是專門 WSN，但它把 routing 做成 packet-centric decision，並用 relational features 讓 policy 泛化到不同拓樸。可當作 future DRL routing 的參考。 |
| [A Deep Reinforcement Learning-based Adaptive Charging Policy for WRSNs](https://arxiv.org/abs/2208.07824) | 2022 | 可充電 WSN 的 mobile charger 決策 | DRL | mobile charger 根據目前網路狀態決定下一個要充電的 sensor。若專案未來加入 recharge/mobile charger，可參考它的 state 與 action 設計。 |
| [Optimal k-Coverage Charging Problem](https://arxiv.org/abs/1901.09129) | 2019 | 在維持 k-coverage 下安排充電路徑 | Dynamic programming + Deep Q-learning | 先證明問題困難，再用 Deep Q-learning 處理大規模情境。和我們的 target coverage 很相關，可參考「coverage constraint + energy action」的設計。 |
| [Multi-Task Lifelong Reinforcement Learning for Wireless Sensor Networks](https://arxiv.org/abs/2506.16254) | 2025 | 動態環境下的 data transmission 與 energy harvesting | Lifelong RL | 讓 WSN 在環境變化時重用過去任務經驗，加快適應。適合長期運行、不同 dataset/round 間希望共享經驗的設定。 |
| [Cross-Domain Lifelong Reinforcement Learning for Wireless Sensor Networks](https://arxiv.org/abs/2508.17852) | 2025 | 跨任務/跨 domain 的 energy-efficient WSN design | Lifelong RL | 處理 EH 條件、網路規模、traffic rate 改變時的快速 policy adaptation。可參考到多場景訓練，而不是每次重新學。 |
| [Active management of battery degradation in wireless sensor network using deep reinforcement learning for group battery replacement](https://arxiv.org/abs/2503.15865) | 2025 | WSN 電池退化與群組更換排程 | DRL | 用 DRL 控制 duty cycle，讓多個 sensor 的電池壽命更同步，降低維護成本。可參考 duty-cycle control 或 long-term lifetime objective。 |
| [Reinforcement Learning Based Whittle Index Policy for Scheduling Wireless Sensors](https://arxiv.org/abs/2601.01179) | 2026 | sensor transmission scheduling 與資料新鮮度 | Q-learning + Whittle Index | 用 Age of Incorrect Information 取代只看 AoI，讓系統選擇最有價值的 sensor 傳輸。可參考在 scheduling fitness 中加入資料價值或重要性。 |
| [Joint link scheduling and power allocation in imperfect and energy-constrained underwater wireless sensor networks](https://arxiv.org/abs/2508.07679) | 2025 | UWSN link scheduling 與 power allocation | Deep MARL / DQN | 把 link scheduling 與 power allocation 做成 Dec-POMDP，處理水下通道、能源限制、節點故障。可參考多目標 routing + power action。 |
| [Traffic Load-Aware Resource Management Strategy for Underwater Wireless Sensor Networks](https://arxiv.org/abs/2508.08555) | 2025 | UWSN traffic-aware resource management | Deep MARL | 用鄰居 overhear information 補足 partial observation，處理 link scheduling、power、rate。可參考 routing 時把 forwarding load 放進 state/reward。 |

## 對目前 2023wsn 的啟發

目前比較適合先實作的是「NSGA-II / SNSGAII 做 scheduler，Q-learning 做 next-hop」這條路線，原因是：

1. WSN 的 schedule/coverage 是全域組合問題，NSGA-II 比 Q-learning 更適合先找出一批 Pareto 解。
2. next-hop 是離散動作，且每次 routing decode 都有候選節點集合，很適合用 Q-learning 或 SARSA。
3. reward 可以直接參考上面的 routing 論文：剩餘能量、到 BS 的距離進展、forwarding load、route capacity、連通失敗懲罰。
4. mobile sensor relocation 可以之後再做 MARL/DRL，因為 action space 會變成移動方向或座標，實作成本比 next-hop 高。
5. 若未來要讓學習跨 dataset 或跨 round 保留經驗，可以參考 lifelong RL，把 Q-table 或 policy 保留下來，而不是每次演算法重跑都清空。

## 建議實驗組合

- `TargetCodingState (priority coding) + greedy routing`
- `TargetCodingState (priority coding) + RQLearning routing`
- `NSGAII + greedy routing`
- `SNSGAII + RQLearning routing`
- `SE-TS + greedy routing`
- `SE-TS + RQLearning routing`

比較指標可以先用目前已有的 fitness：總剩餘能量、最低剩餘能量、coverage，再補 routing 相關指標：disconnect count、平均 hop、forwarding load variance、network lifetime。

## RL Hyper-Heuristic 與方法論補充

這一節補充本次設計新研究方向時查到、對「強化式學習為主、超啟發式為輔」比較重要的論文。它們多數不是 WSN 論文，但方法論很適合轉成目前專案的 `CodingState` operator selection、destroy/repair、SA-SETS/RIME/NSGA-II 混合搜尋控制器。

| 題名 | 年份 | 主要問題 | 方法 | 簡略說明與可參考方向 |
| --- | --- | --- | --- | --- |
| [Recent Advances in Selection Hyper-heuristics](https://doi.org/10.1016/j.ejor.2019.07.073) | 2020 | selection hyper-heuristic 綜述 | Hyper-heuristic survey | 系統整理「選擇低階 heuristic」的框架、特徵、acceptance criterion 與線上/離線學習。可作為 `RL-HH-SETS` 的方法論基礎：RL 不直接產生 routing，而是依目前 WSN 狀態選擇 coverage repair、routing repair、energy mutation、mobility repair 等 operator。 |
| [A Reinforcement Learning Hyper-Heuristic in Multi-Objective Single Point Search with Application to Structural Fault Identification](https://arxiv.org/abs/1812.07958) | 2018 | 多目標最佳化中的 heuristic selection | RL hyper-heuristic + MOSA/R | 使用 domination amount、crowding distance、hypervolume 等資訊讓 RL 選擇低階 heuristic。可參考到 `SNSGAII` / `BaseNSGAII`：把 WSN 的 coverage、energy、min remaining energy 視為多目標，讓 RL 選擇修復或擾動 operator。 |
| [A new Hyper-heuristic based on Adaptive Simulated Annealing and Reinforcement Learning for the Capacitated Electric Vehicle Routing Problem](https://arxiv.org/abs/2206.03185) | 2022 | 電動車路徑規劃與充電限制 | RL + Adaptive Simulated Annealing hyper-heuristic | 用 multi-armed bandit / RL 選擇低階 neighborhood，並用 adaptive SA 控制接受準則。可借鏡成 WSN 版本：RL 選 operator，SA/SETS 負責接受較差但有潛力的 `CodingState`，避免早熟收斂。 |
| [Policy-Based Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems](https://arxiv.org/abs/2601.11189) | 2026 | job-shop scheduling 的 dispatching rule selection | Policy-based DRL hyper-heuristic | 讓 policy 根據系統狀態動態切換 scheduling rules，並提出 action prefiltering 與 commitment mechanism。可轉成 WSN：先過濾不可用 operator，例如沒有 uncovered target 時不選 coverage repair，再限制同一 operator 至少使用若干步，降低搜尋震盪。 |
| [Hyper-heuristics: An Emerging Direction in Modern Search Technology](https://link.springer.com/chapter/10.1007/0-306-48056-5_16) | 2003 | hyper-heuristic 基礎方法 | Heuristics to choose/generate heuristics | 經典方法論，定義 hyper-heuristic 是在 heuristic space 搜尋，而不是 solution space 搜尋。可用來明確區分本專案新方法與一般 WSN metaheuristic：我們不是換一個新自然啟發式名稱，而是讓 RL 學習如何組合既有 WSN-specific operator。 |

## 2026-07-08 補充：值得深入研究的 RL + Hyper-Heuristic / LNS 論文

這一節是針對「還沒有明顯直接套用在本專案 WSN/MWSN coverage-routing-scheduling-energy joint optimization」但很適合轉移過來的方法。初步搜尋 `hyper-heuristic wireless sensor network`、`adaptive large neighborhood search wireless sensor network`、`learning large neighborhood search wireless sensor network` 沒找到強烈直接對應的 WSN 文獻，因此這裡的重點是從 combinatorial optimization / routing / integer programming 借方法，再改造成 `CodingState` 層級的 WSN 演算法。

| 題名 | 年份 | 主要問題 | 方法 | 對 2023wsn 的可研究方向 |
| --- | --- | --- | --- | --- |
| [Graph Reinforcement Learning for Operator Selection in the ALNS Metaheuristic](https://arxiv.org/abs/2302.14678) | 2023 | ALNS 中的 operator selection | GNN + DRL + ALNS | 把 ALNS 的 destroy/repair operator 選擇建成 MDP，讓 policy 根據當前 solution 狀態選 operator。可轉成 WSN：state 用 uncovered target ratio、route disconnected count、energy variance、recent improvement；action 選 coverage destroy/repair、routing repair、energy balance repair、mobile relocation repair。這比普通 Q-routing 更有新意，因為 RL 控制的是搜尋過程，不是封包下一跳。 |
| [Online Control of Adaptive Large Neighborhood Search using Deep Reinforcement Learning](https://arxiv.org/abs/2211.00759) | 2022 | ALNS 參數與接受準則控制 | DRL-controlled ALNS | DRL 同時選 operator、調參數、控制 acceptance criterion。可轉成 WSN：動作不只選 repair operator，也選 repair intensity、SA temperature、接受劣解機率。適合設計成 `RL-ALNS-SETS`，用來改善目前 `SA_SETS` / `RLHH_SETS` 只選單步 operator 的限制。 |
| [Learning Large Neighborhood Search Policy for Integer Programming](https://arxiv.org/abs/2111.03466) | 2021 | Integer programming 的 LNS destroy policy | Actor-critic RL + LNS | RL 學習選哪些變數要釋放/重最佳化。可對應到 `CodingState.code`：學習選哪些 target-sensor assignment、routing priority 或 schedule bit 要 destroy，再用既有 coverage/routing/energy repair 重建。這很適合做「局部重建完整 WSN 解」，而不是微調 reward。 |
| [Learning a Large Neighborhood Search Algorithm for Mixed Integer Programs](https://arxiv.org/abs/2107.10201) | 2021 | MIP 的 learned neighborhood selection | Imitation learning + Neural LNS | 用 learned policy 選擇下一個 large neighborhood，再交給求解器修復。對本專案可改成：用 heuristic/oracle trace 產生訓練資料，讓模型學「什麼情況該修 coverage、routing、energy 或 movement」。如果未來要做 supervised pretraining + RL fine-tuning，這篇很值得看。 |
| [Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem](https://arxiv.org/abs/1911.09539) | 2019 | CVRP / SDVRP 的 learned repair | Neural LNS + attention repair | 在 LNS 框架中加入 learned repair heuristic，效果比手寫 LNS 更好。可借鏡到 WSN：destroy 高負載 relay、低能量 sensor 覆蓋區、斷線 route 後，用 learned repair 重建 target-to-sensor assignment 與 routing priority。 |
| [Learning Combinatorial Optimization Algorithms over Graphs](https://arxiv.org/abs/1704.01665) | 2017 | 圖上 NP-hard 問題的 learned greedy algorithm | Graph embedding + Q-learning | 用圖嵌入與 Q-learning 逐步建構解。可作為「Graph RL constructor for CodingState」的基礎：node 是 target/sensor/base station，edge features 是 distance/connect/coverage relation，action 是把某個 target 指派給候選 sensor 或選擇修復某條 route。 |
| [Graph Reinforcement Learning for Combinatorial Optimization: A Survey and Unifying Perspective](https://arxiv.org/abs/2404.06492) | 2024 | Graph RL for combinatorial optimization 綜述 | Survey | 對 WSN 很有幫助，因為 WSN 天然是 graph/geometry + process optimization。可用來支持論文動機：我們不是把 DRL 直接套 routing，而是把 WSN joint optimization 建成 fixed-graph process optimization。 |
| [Constrained Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/2006.11984) | 2020 | 有限制式的組合最佳化 | Constrained MDP + RL | WSN 解常有硬限制：coverage 不完整、route disconnect、cap 不足、energy 不可負。這篇可支撐 reward/penalty 設計，把 infeasible state 當 constraint violation，而不是只在 fitness 裡加權懲罰。 |
| [Adaptive Operator Selection Based on Dynamic Thompson Sampling for MOEA/D](https://arxiv.org/abs/2004.10874) | 2020 | 多目標演化演算法的 operator selection | Dynamic Thompson Sampling / Bandit AOS | 把 reproduction operator 當 bandit arm，動態更新 reward distribution。可轉成 `SNSGAII` 或 `NSGAII` 的 WSN 版本：根據 coverage/energy/min-energy Pareto 改善量，選擇 crossover、mutation、repair、local search。比固定 mutation rate 更有研究價值。 |
| [Deep Reinforcement Learning Based Parameter Control in Differential Evolution](https://arxiv.org/abs/1905.08006) | 2019 | DE 的 mutation strategy / parameter control | DDQN adaptive operator control | 雖然不是 WSN，但方法很適合「學習控制 metaheuristic 參數」。可轉成本專案：RL 控制 `SA_SETS` 的 mutation intensity、repair ratio、routing repair probability，或控制 `RIME/NSGA` 的搜尋策略。 |
| [HGFF: A Deep Reinforcement Learning Framework for Lifetime Maximization in Wireless Sensor Networks](https://arxiv.org/abs/2407.07747) | 2024 | mobile sink path planning 以最大化 WSN lifetime | Heterogeneous GNN + DRL | 這是 WSN 文獻，但處理的是 mobile sink path，不是本專案的 coverage-routing-scheduling joint CodingState。可借 state representation：sensor/site heterogeneous graph、static/dynamic energy feature、attention aggregation。可用來支持未來 Graph RL 版本，但要明確區分我們不是規劃 sink path。 |

### 對目前新演算法設計的補充判斷

1. 若要避免落入「普通 Q-learning routing」的老題目，最有價值的方向是 RL hyper-heuristic：讓 RL 選擇修復/擾動/移動/局部搜尋 operator，而不是選下一跳。
2. `CodingState` 和 `TargetCodingState` 已經是天然的 solution representation；低階 heuristic 可以包裝成對 `code`、`sch`、`rou`、`use`、`cap` 的可控修改。
3. `SA_SETS`、`SETSv1`、`SRIME`、`SNSGAII` 都可以成為底層搜尋器或 operator 來源，RL 高階策略負責在不同網路狀態下切換。
4. 方法章節可以清楚寫成「state feature extraction -> operator selection -> WSN-specific repair -> acceptance -> reward update」，比單純換 reward function 更有研究差異。

## 近期超啟發式演算法與 WSN 參考論文

這一節整理前面另外查到的近期 metaheuristic / nature-inspired optimization 論文。這些不一定都是 WSN 論文，有些是新演算法本身，有些是已經用在 WSN 的參考案例。用途是幫助後續決定哪些演算法值得改成 `Base*` 模板，或改成 `S* / R*` 的 WSN 版本。

| 題名 | 年份 | 類型 | 簡略說明與可參考方向 |
| --- | --- | --- | --- |
| [A survey on pioneering metaheuristic algorithms between 2019 and 2024](https://arxiv.org/abs/2501.14769) | 2024 | Metaheuristic survey | 整理 2019-2024 間大量新型超啟發式演算法，並用 citation、應用範圍、程式可取得性、參數數量、避免早熟收斂等角度比較。可用來挑選要不要導入 2023wsn 的候選演算法。 |
| [Nature-Inspired Algorithms for Wireless Sensor Networks: A Comprehensive Survey](https://arxiv.org/abs/2101.10453) | 2020 | WSN survey | 專門整理 nature-inspired algorithms 在 WSN 覆蓋問題上的應用，也提到 optimal coverage、能源消耗、redundant sensing。可作為判斷「哪些演算法已經常用在 WSN」的背景資料。 |
| [Ecological Cycle Optimizer: A novel nature-inspired metaheuristic algorithm for global optimization](https://arxiv.org/abs/2508.20458) | 2025 | 新 metaheuristic | ECO 用 ecosystem 中 producer、consumer、decomposer 的能量流和物質循環來設計搜尋策略。可嘗試對應到 WSN：producer 產生覆蓋解、consumer 改善能量與路由、decomposer 修復不可行解。 |
| [A modified RIME algorithm with covariance learning and diversity enhancement for numerical optimization](https://arxiv.org/abs/2509.09529) | 2025 | RIME 改良版 | MRIME-CD 針對 RIME 容易多樣性下降、陷入 local optimum 的問題，加入 covariance learning、多樣性增強與停滯更新策略。可作為目前 `SRIME` 後續優化版本的參考。 |
| [TERIME: An improved RIME algorithm with enhanced exploration and exploitation for robust parameter extraction of photovoltaic models](https://arxiv.org/abs/2407.18986) | 2024 | RIME 應用與改良 | 把 RIME 改成更穩定的 exploration/exploitation 結構，用在 PV model parameter extraction。可參考它如何把差分突變、鄰域策略加進 RIME，未來可改成 sensing radius repair 的強化版。 |
| [Duck swarm algorithm: theory, numerical optimization, and applications](https://arxiv.org/abs/2112.13508) | 2021/2024 | 新 metaheuristic | DSA 模擬 duck swarm 找食物與覓食行為，分別對應 exploration 和 exploitation。這篇本身已有 node deployment 應用，因此若要追求「還沒用在 WSN」的新意，DSA 不是優先選項，但可作比較 baseline。 |
| [Balancing the trade-off between cost and reliability for wireless sensor networks: a multi-objective optimized deployment method](https://arxiv.org/abs/2207.09089) | 2022 | WSN deployment | 使用 competitive multi-objective marine predators algorithm 做 WSN node deployment，目標是在部署成本、感測可靠性、網路可靠性間取平衡。這代表 MPA 類方法已經用於 WSN，所以不適合作為全新演算法主軸，但適合作比較。 |
| [A new approach for solving global optimization and engineering problems based on modified Sea Horse Optimizer](https://arxiv.org/abs/2402.14044) | 2024 | SHO 改良版 | 改良 Sea Horse Optimizer，加入 local search 與更強 exploitation，用 benchmark 和工程問題驗證。可參考它的 local search 設計，用在 WSN repair 或 radius 微調。 |
| [A Memetic Walrus Algorithm with Expert-guided Strategy for Adaptive Curriculum Sequencing](https://arxiv.org/abs/2506.13092) | 2025 | Walrus optimizer 應用/改良 | 這篇不是 WSN，而是把 Walrus Optimizer 改成 memetic version 用於 curriculum sequencing。可參考它的 expert-guided strategy 和 aging mechanism，未來若做 `SWSO` 或排程 repair，可借用避免 local optimum 的想法。 |

## 對目前新演算法選擇的補充判斷

目前比較有研究價值的方向仍是：

1. `SRIME`：適合 sensing radius repair，尤其是把離散 level 修成更貼近實際需求的半徑。
2. `SECO` 或 `ECO-WSN`：適合做 schedule + repair 的混合搜尋，論文包裝空間比較大。
3. `INFO-WSN`：可以考慮，但目前需要再補原始論文與穩定參考來源後再實作。

不優先作為主方法的方向：

1. MPA 類方法：已經有 WSN deployment 應用，較適合作 baseline。
2. DSA：原論文已包含 WSN node deployment 應用，研究新意比較弱。
3. 單純換新的 metaphor algorithm：若沒有 WSN-specific decoder / repair，容易變成只換演算法名字，貢獻不夠清楚。

## 2026-07-11 補充：非超啟發式、非強化式學習的 WSN 演算法

這一節補查不在前面清單的論文，且排除以 metaheuristic、hyper-heuristic、RL/DRL/MARL 為主的方法。重點是目前專案已有的 target coverage、可調 sensing range、active/sleep scheduling、multi-hop routing、energy balance 與 network lifetime。這些方法特別適合當精確 baseline，或作為下一篇研究的主框架。

| 題名 | 年份 | 方法 | 主要問題與可參考方向 |
| --- | --- | --- | --- |
| [A comprehensive survey on linear programming and energy optimization methods for maximizing lifetime of wireless sensor network](https://doi.org/10.1007/s10791-024-09454-5) | 2024 | LP/MILP survey | 整理 2017-2023 年間 LP、energy optimization、load balancing 的 WSN 研究，可作非 metaheuristic 方法的文獻入口。 |
| [Coverage and connectivity based lifetime maximization with topology update for WSN in smart grid applications](https://doi.org/10.1016/j.comnet.2022.108940) | 2022 | Multi-period 0-1 MIP | 每個 period 依剩餘能量重建 topology 與 transmission path，同時維持 coverage 與 BS connectivity。很接近目前逐 round 更新 energy、schedule、routing 的流程。 |
| [An exact approach to extend network lifetime in a general class of wireless sensor networks](https://doi.org/10.1016/j.ins.2017.12.028) | 2018 | Column generation + branch-and-cut | 以 role 表示 sensing range、direction、communication mode 的組合，再作大規模 LP 分解。可把 role 對應成目前 sensor 的關閉、感測 level、source 與 relay。 |
| [Exact approaches for lifetime maximization in connectivity constrained wireless multi-role sensor networks](https://doi.org/10.1016/j.ejor.2014.08.013) | 2015 | Column generation + Benders / CP | 將 cover scheduling 放在 master problem，pricing 使用 Benders branch-and-cut 或 constraint programming，並區分 source 與純 relay。和目前 Coverage/Routing/Energy service 的分層最吻合。 |
| [A column generation approach to extend lifetime in wireless sensor networks with coverage and connectivity constraints](https://doi.org/10.1016/j.cor.2013.11.001) | 2014 | Column generation + ILP | 同時處理 full coverage 與 alpha partial coverage；以 CG 建立 cover，再以 ILP 驗證最適性。可當 connected cover scheduling 的直接參考。 |
| [An exact approach for maximizing the lifetime of sensor networks with adjustable sensing ranges](https://doi.org/10.1016/j.cor.2012.04.001) | 2012 | Column generation | 同時處理連續與離散 sensing range，和目前 `LEVEL_RANGE` 最直接相關；可作 sensing radius 與 schedule 聯合最佳化的基礎。 |
| [A column generation based heuristic for sensor placement, activity scheduling and data routing in wireless sensor networks](https://doi.org/10.1016/j.ejor.2010.05.020) | 2010 | MILP + column generation | 聯合處理 sensor/sink placement、data flow、activity schedule、coverage、energy 與 budget；可核對目前 joint model 是否需補 flow conservation。 |
| [A novel differentiated coverage-based lifetime metric for wireless sensor networks](https://doi.org/10.1016/j.adhoc.2024.103636) | 2024 | 0-1 MIP + 新 lifetime metric | 將每個 target 的 coverage requirement 納入 lifetime，而非只用 FND/LND。可將 target importance 或不同 beta coverage 加入目前 fitness/constraint。 |
| [Partial coverage optimization under network connectivity constraints in heterogeneous sensor networks](https://doi.org/10.1016/j.comnet.2022.108928) | 2022 | Mathematical optimization | 同時使用全域 alpha coverage 與逐 target beta coverage；可避免 partial coverage 只看總覆蓋比例、忽略關鍵 target。 |
| [Maximizing network lifetime using coverage sets scheduling in wireless sensor networks](https://doi.org/10.1016/j.adhoc.2019.102037) | 2020 | ILP + greedy + approximation | 定義 Maximum Coverage Sets Scheduling (MCSS)，並提供 greedy 和 approximation algorithm。適合當非隨機、可重現且具理論分析的 scheduling baseline。 |
| [Optimally Approximating the Coverage Lifetime of Wireless Sensor Networks](https://arxiv.org/abs/1307.5230) | 2013 | Approximation algorithm | 對 coverage lifetime 提供 approximation 結果；二維圓形 sensing case 可達 `(1+epsilon)` approximation。適合補強理論型 benchmark。 |
| [Energy and throughput aware adequate routing for wireless sensor networks using integrated game theory method](https://doi.org/10.1038/s41598-024-71902-5) | 2024 | Cooperative game theory | 以 cooperative game theory 處理能量與 throughput 的 routing trade-off，可作 non-RL next-hop routing baseline。 |
| [k-Coverage in Three-Dimensional Wireless Sensor Networks Using a Game Theoretical Approach](https://doi.org/10.1109/MASS66014.2025.00108) | 2025 | Game-theoretic scheduling | 用分散式賽局決策達成 k-coverage 與延長 lifetime。場景是 3D，但可參考其 scheduling utility 設計。 |
| [Wireless Link Scheduling via Graph Representation Learning: A Comparative Study of Different Supervision Levels](https://arxiv.org/abs/2110.01722) | 2021 | Supervised/unsupervised/self-supervised GNN | 雖非 WSN 專屬，仍可作「從精確解學習快速 link scheduler」的依據；不是 RL，但需要先有 MILP/CG 解當標註資料。 |

### 對目前專案的優先判斷

最值得優先做的不是再替換一個自然隱喻演算法，而是建立 **Column Generation + Benders / Constraint Programming** 的精確或 matheuristic 框架：

1. 將可行的 `CodingState`（active sensor、sensing level、routing tree）視為一個 connected cover/column。
2. master problem 決定各 cover 的啟用時間，最大化 lifetime，並限制每顆 sensor 的 energy budget。
3. pricing problem 找新的低成本 connected cover：coverage/sensing 部分選 sensor 與 level；routing 部分建立到 BS 的 flow/tree。
4. 當 coverage schedule 不能連通或 energy 超限時，以 Benders cut 排除，而不是只在 fitness 加 penalty。
5. 現有 `CCS`、greedy、`NSGAII`、`SNSGAII`、`SA_SETS`、`SRIME` 的解可作 initial columns 或 pricing heuristic。

這條路線和目前 `CoverageService`、`RoutingService`、`EnergyService` 的分工直接對應；在小型 instance 可取得 exact solution 或 optimality gap，讓現有演算法不只互相比較，也能量化離最佳解有多遠。

### 建議實作順序

1. 先建立單 round MILP baseline：binary active、sensing level、target assignment、directed flow；先以 max-min remaining energy 或 min-max energy cost 為目標。
2. 用 epsilon-constraint 產生 coverage/lifetime/energy-balance Pareto front，避免只用加權 fitness。
3. 由小型 MILP 驗證 service 計算一致性，報告 feasibility、objective gap、runtime。
4. 再做 multi-round 0-1 MIP 或 rolling horizon，處理每 round energy depletion 和 topology update。
5. 規模再大時擴充為 CG-Benders：現有 heuristic 僅負責 warm start/pricing，而非論文主方法。

不建議先以單純 MST、shortest-path、PEGASIS 變體作主題，因為它們雖適合作 baseline，卻難同時處理 adjustable range、coverage、routing capacity 與 lifetime；GNN 也應等精確解累積後再做 warm start，而不宜直接取代 feasibility solver。
