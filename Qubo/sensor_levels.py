"""區域感測等級子問題，以及把它寫成 QUBO 的溫度計／one-hot 編碼。

子問題：固定區域外 sensor 的等級，只重新決定區域內 k 顆 sensor 的等級，
讓「區域外蓋不到、區域內蓋得到」的 target 全部被覆蓋，並使目標最小。

所有目標都寫成「每顆 sensor 的等級表」加上「群組平方項」：

    F = Σ_j linear[j, L_j] + Σ_g w_g · (offset_g + Σ_j group[j, L_j, g])²

等級 0（關閉）的表值一律為 0。三種目標：

    ``burden``  ：linear = (感測耗能 + 開啟成本) / 剩餘電量，沒有群組（第一版）。
    ``fitness`` ：貼近 fitness v2，但把「送回 BS 的整條路徑耗能」都算在
                  sensor 自己身上（已知會高估換 sensor 的好處，保留作對照）。
    ``routing`` ：同樣貼近 fitness v2，但固定目前的路由樹，sensor 開啟時
                  自己的資料沿路徑加到每一顆中繼 sensor 的耗能上，並禁止
                  關閉正在幫別人轉送的 sensor。路由樹不變時，耗能與
                  ``calculate_routing_cost`` 完全一致。

fitness v2 的 0.7 項只看最差 target，QUBO 無法寫 max，改以
w_t ∝ (目前耗損 / 最差耗損)^q 加權的平方和近似。F 最後除以「單顆 sensor
的最大邊際成本」，讓懲罰可以用固定數值。這仍是代理目標；解碼後必須交回
Problem 建路由並評估真正的 fitness。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Qubo.model import QuboModel
from State.PriorityEncoding import PriorityEncoding

# 目標縮放後，任一顆 sensor 的邊際成本 ≤ 1；丟掉一個 target 最多省下一顆
# sensor 的成本，懲罰取 2 以確保違反任何限制都不划算。
DEFAULT_PENALTY = 2.0
# target 權重隨「目前耗損 ÷ 最差耗損」的次方遞減；次方越高，越集中在最差的
# 幾個 target 上。
DEFAULT_FOCUS_POWER = 4.0
OBJECTIVES = ("burden", "fitness", "routing")


@dataclass
class LevelSubproblem:
    """一個區域的等級子問題；區域內陣列都以區域位置 j 索引。"""

    sensors: np.ndarray            # 區域內可最佳化的 sensor id
    option_counts: np.ndarray      # 每顆 sensor 的選項數（含 0 = 關閉）
    min_level: np.ndarray          # 每顆 sensor 的最低等級（中繼 sensor 為 1）
    linear: np.ndarray             # (k, L) 線性表；不存在的選項為 inf
    group: np.ndarray              # (k, L, G) 各等級對群組的貢獻
    group_offset: np.ndarray       # (G,) 群組常數
    group_weight: np.ndarray       # (G,) 群組平方項權重 w_g
    group_targets: np.ndarray      # (G,) 每個群組對應的 target id
    parents: np.ndarray            # (k,) routing 目標假設的下一跳；其他目標為 -1
    scale: float                   # 目標除以此值
    required_targets: np.ndarray   # 必須由區域覆蓋的 target id
    unreachable_targets: np.ndarray  # 區域外蓋不到、區域內也蓋不到
    lmin: np.ndarray               # (len(required), k)；蓋不到為 -1
    base_levels: np.ndarray        # 完整的原始等級（長度 = SENSOR_NUMBER）
    objective_name: str

    @property
    def size(self):
        return len(self.sensors)

    def covered(self, region_levels):
        """回傳每個 required target 是否被區域內的等級覆蓋。"""
        region_levels = np.asarray(region_levels)
        reach = (self.lmin >= 1) & (self.lmin <= region_levels[None, :])
        return reach.any(axis=1)

    def feasible_batch(self, region_levels):
        """(N, k) 的等級是否全覆蓋且不低於最低等級。"""
        levels = np.atleast_2d(np.asarray(region_levels, dtype=int))
        reach = (self.lmin[None] >= 1) & (self.lmin[None] <= levels[:, None, :])
        covered = reach.any(axis=2).all(axis=1)
        return covered & np.all(levels >= self.min_level[None, :], axis=1)

    def is_feasible(self, region_levels):
        return bool(self.feasible_batch(region_levels)[0])

    def objective_batch(self, region_levels, scaled=True):
        """一次計算多組等級 (N, k) 的目標值。"""
        levels = np.atleast_2d(np.asarray(region_levels, dtype=int))
        rows = np.arange(self.size)[None, :]
        value = self.linear[rows, levels].sum(axis=1)
        if len(self.group_weight):
            load = self.group_offset[None, :] + self.group[rows, levels].sum(axis=1)
            value = value + (load * load) @ self.group_weight
        return value / self.scale if scaled else value

    def objective(self, region_levels):
        """縮放後的目標值（QUBO 在合法且全覆蓋時的能量）。"""
        return float(self.objective_batch(region_levels)[0])

    def region_levels_of(self, levels):
        """從完整等級向量取出區域內的等級。"""
        return np.asarray(levels, dtype=int)[self.sensors].copy()

    def full_levels(self, region_levels):
        """把區域內的新等級寫回完整等級向量（區域外不變）。"""
        levels = self.base_levels.copy()
        levels[self.sensors] = np.asarray(region_levels, dtype=int)
        return levels


def _path_to_bs(problem, state, node):
    """沿 next_hops 走到 BS，回傳經過的 sensor（不含 BS）；斷線回傳 None。"""
    path = []
    seen = set()
    while node != problem.BSID:
        if node < 0 or node >= problem.SENSOR_NUMBER or node in seen:
            return None
        seen.add(node)
        path.append(int(node))
        node = int(state.next_hops[node])
    return path


def _routing_effects(problem, state, sensors, region_mask):
    """固定路由樹時，每顆區域 sensor 開啟後「自己的資料」造成的耗能。

    回傳 ``(parents, own, chains, min_level, usable)``：
    own[j] 是 sensor j 第一跳的耗能；chains[j] 是 [(中繼 id, 耗能), ...]。
    耗能公式與 ``calculate_routing_cost_kernel`` 一致：自己那一跳為
    load × (amp + circuit)，每個中繼為 load × (amp + 2·circuit)。

    尚未接入的 sensor 依 ``build_routes`` 的規則估計下一跳：在已接入的節點
    中取 min(鏈路容量, 路徑剩餘容量) − load 最大者。區域內沒有在轉送的
    sensor 可能被關掉，因此不拿來當下一跳。
    """
    amp = np.asarray(problem.amp_cost, dtype=float)
    circuit = float(problem.circuit_cost)
    generated = np.asarray(problem.generated_load, dtype=float)
    link = np.asarray(problem.link_capacity, dtype=float)
    remaining = np.asarray(state.remaining_capacity, dtype=float).copy()
    # 尚未建過路由的 state 保留預設值 -1；build_routes 永遠視 BS 為無限容量。
    remaining[problem.BSID] = np.inf
    next_hops = np.asarray(state.next_hops, dtype=int)

    active = np.asarray(state.levels) > 0
    routed = np.array(
        [bool(active[i]) and _path_to_bs(problem, state, i) is not None
         for i in range(problem.SENSOR_NUMBER)]
    )
    is_relay = np.zeros(problem.SENSOR_NUMBER, dtype=bool)
    children = next_hops[routed & (next_hops >= 0) & (next_hops < problem.SENSOR_NUMBER)]
    is_relay[children] = True

    # 可當下一跳的節點：BS、區域外已接入的 sensor、區域內的中繼。
    eligible = routed & (~region_mask | is_relay)
    candidates = np.append(np.flatnonzero(eligible), problem.BSID)

    k = len(sensors)
    parents = np.full(k, -1, dtype=int)
    own = np.zeros(k)
    chains = [[] for _ in range(k)]
    usable = np.ones(k, dtype=bool)
    for j, sensor_id in enumerate(sensors):
        load = generated[sensor_id]
        if routed[sensor_id]:
            parent = int(next_hops[sensor_id])
        else:
            bottleneck = np.minimum(link[sensor_id, 1, candidates], remaining[candidates])
            best = int(np.argmax(bottleneck))
            if bottleneck[best] - load < 0:
                usable[j] = False
                continue
            parent = int(candidates[best])
        parents[j] = parent
        own[j] = load * (amp[sensor_id, parent] + circuit)
        if parent != problem.BSID:
            for relay in _path_to_bs(problem, state, parent) or []:
                hop = int(next_hops[relay])
                chains[j].append((relay, load * (amp[relay, hop] + 2.0 * circuit)))
    min_level = is_relay[sensors].astype(int)
    return parents, own, chains, min_level, usable


def build_level_subproblem(
    problem,
    state,
    region,
    objective="routing",
    include_open_cost=True,
    focus_power=DEFAULT_FOCUS_POWER,
):
    """依目前已解碼（含路由）的 ``state`` 建立 ``region`` 的子問題。

    電量耗盡、無法送回 BS、或（routing 目標下）找不到下一跳的 sensor 不能
    被最佳化，會留在區域外並維持原等級。
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    levels = np.asarray(state.levels, dtype=int)
    coverage = np.asarray(problem.coverage_table) > 0
    option_counts_all = np.asarray(problem.radius_option_counts, dtype=int)
    energy = np.asarray(problem.energy, dtype=float)
    sensing_costs = np.asarray(problem.sensing_costs, dtype=float)

    # 開啟成本沿用 PriorityEncoding 的回傳成本快取（只和位置有關）。
    delivery_cost = PriorityEncoding._delivery_cost(problem)
    open_cost = np.asarray(problem.generated_load, dtype=float) * delivery_cost

    region = np.asarray(region, dtype=int)
    sensors = region[(energy[region] > 0) & np.isfinite(open_cost[region])]
    min_level = np.zeros(len(sensors), dtype=int)
    parents = np.full(len(sensors), -1, dtype=int)
    if objective == "routing" and len(sensors):
        region_mask = np.zeros(problem.SENSOR_NUMBER, dtype=bool)
        region_mask[sensors] = True
        parents, own, chains, min_level, usable = _routing_effects(
            problem, state, sensors, region_mask
        )
        sensors, own, min_level = sensors[usable], own[usable], min_level[usable]
        parents = parents[usable]
        chains = [chain for chain, keep in zip(chains, usable) if keep]
    if len(sensors) == 0:
        raise ValueError("region has no usable sensor")
    region_mask = np.zeros(problem.SENSOR_NUMBER, dtype=bool)
    region_mask[sensors] = True

    outside_ids = np.flatnonzero(~region_mask)
    covered_outside = coverage[:, outside_ids, levels[outside_ids]].any(axis=1)
    option_counts = option_counts_all[sensors]
    coverable = coverage[:, sensors, option_counts - 1].any(axis=1)
    required = np.flatnonzero(~covered_outside & coverable)
    unreachable = np.flatnonzero(~covered_outside & ~coverable)

    k = len(sensors)
    level_count = int(option_counts.max())
    invalid = np.arange(level_count)[None, :] >= option_counts[:, None]

    # effect[j][l]：sensor j 開到等級 l 時，加在各 sensor 上的耗能 {id: 耗能}。
    effects = []
    for j, sensor_id in enumerate(sensors):
        rows = [{}]
        for level in range(1, int(option_counts[j])):
            change = {int(sensor_id): float(sensing_costs[sensor_id, level])}
            if objective == "routing":
                change[int(sensor_id)] += own[j]
                for relay, value in chains[j]:
                    change[relay] = change.get(relay, 0.0) + value
            elif include_open_cost:
                change[int(sensor_id)] += float(open_cost[sensor_id])
            rows.append(change)
        effects.append(rows)

    if objective == "burden":
        linear = np.zeros((k, level_count))
        for j, sensor_id in enumerate(sensors):
            for level in range(1, int(option_counts[j])):
                linear[j, level] = sum(effects[j][level].values()) / energy[sensor_id]
        group = np.zeros((k, level_count, 0))
        offset = np.zeros(0)
        weight = np.zeros(0)
        group_targets = np.zeros(0, dtype=int)
    else:
        linear, group, offset, weight, group_targets = _fitness_tables(
            problem, state, sensors, option_counts, effects, objective, focus_power
        )
    linear[invalid] = np.inf

    # 覆蓋隨等級嵌套，所以只需記錄「至少要開到哪一級」。
    lmin = np.full((len(required), k), -1, dtype=int)
    for r, target_id in enumerate(required):
        for j, sensor_id in enumerate(sensors):
            hits = np.flatnonzero(coverage[target_id, sensor_id, 1:option_counts[j]])
            if len(hits):
                lmin[r, j] = int(hits[0]) + 1

    sub = LevelSubproblem(
        sensors=sensors,
        option_counts=option_counts,
        min_level=min_level,
        linear=linear,
        group=group,
        group_offset=offset,
        group_weight=weight,
        group_targets=group_targets,
        parents=parents,
        scale=1.0,
        required_targets=required,
        unreachable_targets=unreachable,
        lmin=lmin,
        base_levels=levels.copy(),
        objective_name=objective,
    )
    sub.scale = _marginal_scale(sub)
    return sub


def _fitness_tables(problem, state, sensors, option_counts, effects, objective, focus_power):
    """fitness／routing 目標的線性表與群組表。

    群組是「耗能會被區域改變的 sensor」周圍的每個 target；d_t 為
    (固定耗能 + 區域造成的耗能) / t 周圍的電量總和。固定耗能取目前 state
    的真實耗能，扣掉區域 sensor 目前自己造成的部分。
    """
    energy = np.asarray(problem.energy, dtype=float)
    mask = np.asarray(problem.target_sensor_mask, dtype=float)
    service = problem.fitness_service
    energy_weight = float(getattr(service, "w1", 0.3))
    target_weight = float(getattr(service, "w2", 0.7))
    total_energy = float(np.sum(np.maximum(energy, 0.0)))

    radii = problem.state_radius(state)
    current_cost = np.asarray(
        problem.calculate_total_cost(state, sensing_radii=radii), dtype=float
    )
    levels = np.asarray(state.levels, dtype=int)
    constant = current_cost.copy()
    if objective == "routing":
        # 路由樹固定：扣掉區域 sensor 目前自己的感測與資料所造成的耗能，
        # 中繼幫別人轉送的部分保留。
        for j, sensor_id in enumerate(sensors):
            for node, value in effects[j][levels[sensor_id]].items():
                constant[node] -= value
    else:
        # 第一版近似：區域 sensor 只保留「幫別人轉送」的部分。
        routing_cost = np.asarray(
            problem.energy_service.calculate_routing_cost(state, sensing_radii=radii),
            dtype=float,
        )
        tx_load = np.asarray(state.tx_load, dtype=float)
        generated = np.asarray(problem.generated_load, dtype=float)
        own_share = np.where(
            tx_load > 0, routing_cost * generated / np.maximum(tx_load, 1e-12), 0.0
        )
        constant[sensors] = np.maximum(routing_cost - own_share, 0.0)[sensors]
    constant = np.maximum(constant, 0.0)

    target_energy = mask @ energy
    depletion = (mask @ current_cost) / np.maximum(target_energy, 1e-12)
    worst = float(np.max(depletion)) or 1.0

    touched_sensors = sorted({node for rows in effects for row in rows for node in row})
    groups = np.flatnonzero(mask[:, touched_sensors].any(axis=1))

    k = len(sensors)
    level_count = int(option_counts.max())
    linear = np.zeros((k, level_count))
    group = np.zeros((k, level_count, len(groups)))
    for j in range(k):
        for level in range(1, int(option_counts[j])):
            change = effects[j][level]
            nodes = np.fromiter(change.keys(), dtype=int)
            values = np.fromiter(change.values(), dtype=float)
            linear[j, level] = energy_weight * values.sum() / total_energy
            group[j, level] = (mask[np.ix_(groups, nodes)] @ values) / target_energy[groups]
    offset = (mask[groups] @ constant) / target_energy[groups]
    ratio = depletion[groups] / worst
    weight = target_weight * ratio ** focus_power / (2.0 * worst)
    return linear, group, offset, weight, groups


def apply_with_fixed_routes(problem, state, sub, region_levels):
    """套用區域新等級並保留原本的路由樹；違反鏈路容量時回傳 None。

    關掉的 sensor 拿掉下一跳，開啟的 sensor 接到 routing 目標假設的下一跳
    （原本已接入者不變），其他 sensor 的路由完全不動，因此結果的耗能與
    routing 目標的模型一致。負載與剩餘容量依 ``build_routes_kernel`` 的
    規則重算：每顆 sensor 的資料量加到路徑上每一個節點，剩餘容量為路徑上
    min(鏈路容量 − 負載)；任何接入節點的剩餘容量為負即不可行。
    """
    if sub.objective_name != "routing":
        raise ValueError("fixed routes need the routing objective's parents")
    new = state.copy()
    region_levels = np.asarray(region_levels, dtype=int)
    for j, sensor_id in enumerate(sub.sensors):
        level = int(region_levels[j])
        new.levels[sensor_id] = level
        new.next_hops[sensor_id] = int(sub.parents[j]) if level > 0 else -1

    bs = problem.BSID
    generated = np.asarray(problem.generated_load)
    link = np.asarray(problem.link_capacity, dtype=float)
    new.tx_load = np.zeros_like(np.asarray(state.tx_load))
    new.paths = {int(bs): [int(bs)]}
    routed = []
    for sensor_id in np.flatnonzero(new.levels > 0):
        path = _path_to_bs(problem, new, int(sensor_id))
        if path is None:
            continue  # 原本就斷線的 sensor 維持原狀，交給 fitness 判定
        new.paths[int(sensor_id)] = path + [int(bs)]
        new.tx_load[path] += generated[sensor_id]
        routed.append((len(path), int(sensor_id)))

    remaining = np.full(problem.DEVICE_NUMBER, -1.0)
    remaining[bs] = np.inf
    for _, sensor_id in sorted(routed):
        parent = int(new.next_hops[sensor_id])
        direct = link[sensor_id, new.levels[sensor_id], parent] - new.tx_load[sensor_id]
        remaining[sensor_id] = min(direct, remaining[parent])
        if remaining[sensor_id] < 0:
            return None
    new.remaining_capacity = remaining
    problem.resolve_state_radius(new)
    return new


def _marginal_scale(sub):
    """其他 sensor 全開到最大時，單顆 sensor 從關閉到最大的最大邊際成本。

    各等級的耗能都是非負且隨等級遞增，群組平方項是凸的，所以這是任何
    一步邊際成本的上界。
    """
    top = sub.option_counts - 1
    trials = np.repeat(top[None, :], sub.size + 1, axis=0)
    trials[np.arange(sub.size), np.arange(sub.size)] = 0
    values = sub.objective_batch(trials, scaled=False)
    marginal = float(np.max(values[-1] - values[:-1]))
    return marginal if marginal > 0 else 1.0


def solve_exact(sub, chunk=200_000):
    """列舉區域內所有等級組合，回傳 ``(最佳等級, 目標值)``；無解回傳 (None, inf)。"""
    counts = [int(c) for c in sub.option_counts]
    total = int(np.prod(counts))
    best_levels, best_value = None, np.inf
    for start in range(0, total, chunk):
        index = np.arange(start, min(start + chunk, total))
        levels = np.empty((len(index), sub.size), dtype=int)
        rest = index.copy()
        for j in range(sub.size - 1, -1, -1):
            levels[:, j] = rest % counts[j]
            rest //= counts[j]
        feasible = sub.feasible_batch(levels)
        if not feasible.any():
            continue
        values = sub.objective_batch(levels[feasible])
        position = int(np.argmin(values))
        if values[position] < best_value:
            best_value = float(values[position])
            best_levels = levels[feasible][position].copy()
    return best_levels, best_value


def slack_weights(max_value):
    """有界二進位 slack：各 bit 權重的和剛好等於 ``max_value``。

    例如 max_value = 4 → [1, 2, 1]，可表示 0～4 的每一個整數。
    """
    max_value = int(max_value)
    if max_value <= 0:
        return []
    bits = int(np.floor(np.log2(max_value))) + 1
    weights = [1 << position for position in range(bits - 1)]
    weights.append(max_value - ((1 << (bits - 1)) - 1))
    return weights


class LevelEncoding:
    """把 LevelSubproblem 轉成 QUBO，並把 0/1 解轉回等級。"""

    name = "base"

    def build(self, sub, penalty=DEFAULT_PENALTY):
        """回傳 QuboModel；等級變數之後接著覆蓋限制的 slack 變數。"""
        model = QuboModel()
        self.index = {}
        for j in range(sub.size):
            for level in range(1, int(sub.option_counts[j])):
                self.index[j, level] = model.add_variable(
                    f"{self.name}[s{int(sub.sensors[j])},{level}]"
                )
        self._add_objective(model, sub)
        self._add_validity(model, sub, penalty)
        self._add_must_on(model, sub, penalty)
        self._add_coverage(model, sub, penalty)
        return model

    def _add_objective(self, model, sub):
        """F / scale：各表以編碼的線性式表示，群組項用 add_squared 展開。"""
        for j in range(sub.size):
            for index, value in self._level_terms(sub, j, sub.linear[j]):
                model.add_linear(index, value / sub.scale)
        for g in range(len(sub.group_weight)):
            terms = [
                term
                for j in range(sub.size)
                for term in self._level_terms(sub, j, sub.group[j, :, g])
            ]
            model.add_squared(
                terms, sub.group_offset[g], sub.group_weight[g] / sub.scale
            )

    def _add_coverage(self, model, sub, penalty):
        """每個 required target：penalty · (Σ 覆蓋指示 − 1 − slack)²。

        slack 吸收多出來的覆蓋次數，所以懲罰永遠 ≥ 0，重複覆蓋不會
        變成獎勵；只有完全沒被覆蓋時才會被罰。
        """
        for r, target_id in enumerate(sub.required_targets):
            terms = []
            candidates = 0
            for j in range(sub.size):
                level = int(sub.lmin[r, j])
                if level < 1:
                    continue
                candidates += 1
                terms.extend(self._coverage_terms(sub, j, level))
            for bit, weight in enumerate(slack_weights(candidates - 1)):
                slack = model.add_variable(f"slack[t{int(target_id)},{bit}]")
                terms.append((slack, -weight))
            model.add_squared(terms, -1.0, penalty)

    def _level_terms(self, sub, j, table):
        """等級表 table[等級] 的線性表示 [(index, 係數), ...]；table[0] 須為 0。"""
        raise NotImplementedError

    def _add_validity(self, model, sub, penalty):
        raise NotImplementedError

    def _add_must_on(self, model, sub, penalty):
        """中繼 sensor（最低等級 1）沒開就罰 penalty。"""
        raise NotImplementedError

    def _coverage_terms(self, sub, j, level):
        """「sensor j 的等級 ≥ level」的線性表示 [(index, 係數), ...]。"""
        raise NotImplementedError

    def decode(self, sub, x):
        """回傳 ``(region_levels, valid)``；valid 表示等級位元是否合法。"""
        raise NotImplementedError


class ThermometerEncoding(LevelEncoding):
    """y[j, l] = 1 表示「等級 ≥ l」；合法解是前段全 1、後段全 0。"""

    name = "thermo"

    def _level_terms(self, sub, j, table):
        # 每升一級的增量；合法時加總剛好等於 table[等級]。
        return [
            (self.index[j, level], table[level] - table[level - 1])
            for level in range(1, int(sub.option_counts[j]))
        ]

    def _add_validity(self, model, sub, penalty):
        # penalty · y[l+1] · (1 − y[l])：後面是 1、前面是 0 才罰。
        for j in range(sub.size):
            for level in range(1, int(sub.option_counts[j]) - 1):
                lower = self.index[j, level]
                upper = self.index[j, level + 1]
                model.add_linear(upper, penalty)
                model.add_quadratic(lower, upper, -penalty)

    def _add_must_on(self, model, sub, penalty):
        for j in np.flatnonzero(sub.min_level >= 1):
            model.add_linear(self.index[j, 1], -penalty)
            model.add_offset(penalty)

    def _coverage_terms(self, sub, j, level):
        return [(self.index[j, level], 1.0)]

    def decode(self, sub, x):
        x = np.asarray(x, dtype=int)
        region_levels = np.zeros(sub.size, dtype=int)
        valid = True
        for j in range(sub.size):
            bits = [x[self.index[j, level]] for level in range(1, int(sub.option_counts[j]))]
            leading = 0
            while leading < len(bits) and bits[leading] == 1:
                leading += 1
            # 不合法時從第一個 0 截斷，保留連續的低等級部分。
            if any(bits[leading:]):
                valid = False
            region_levels[j] = leading
        return region_levels, valid


class OneHotEncoding(LevelEncoding):
    """x[j, l] = 1 表示「剛好是等級 l」；全部為 0 代表關閉。"""

    name = "onehot"

    def _level_terms(self, sub, j, table):
        return [
            (self.index[j, level], table[level])
            for level in range(1, int(sub.option_counts[j]))
        ]

    def _add_validity(self, model, sub, penalty):
        # 最多一個 1：任兩個同時為 1 就罰。
        for j in range(sub.size):
            levels = range(1, int(sub.option_counts[j]))
            for level in levels:
                for other in levels:
                    if other > level:
                        model.add_quadratic(
                            self.index[j, level],
                            self.index[j, other],
                            penalty,
                        )

    def _add_must_on(self, model, sub, penalty):
        # penalty · (Σ_l x[j, l] − 1)²：剛好一個等級時為 0。
        for j in np.flatnonzero(sub.min_level >= 1):
            terms = [
                (self.index[j, level], 1.0)
                for level in range(1, int(sub.option_counts[j]))
            ]
            model.add_squared(terms, -1.0, penalty)

    def _coverage_terms(self, sub, j, level):
        return [
            (self.index[j, chosen], 1.0)
            for chosen in range(level, int(sub.option_counts[j]))
        ]

    def decode(self, sub, x):
        x = np.asarray(x, dtype=int)
        region_levels = np.zeros(sub.size, dtype=int)
        valid = True
        for j in range(sub.size):
            chosen = [
                level
                for level in range(1, int(sub.option_counts[j]))
                if x[self.index[j, level]] == 1
            ]
            if len(chosen) > 1:
                valid = False
            # 多選時取最高等級，避免解碼後失去覆蓋。
            region_levels[j] = max(chosen) if chosen else 0
        return region_levels, valid


ENCODINGS = {
    ThermometerEncoding.name: ThermometerEncoding,
    OneHotEncoding.name: OneHotEncoding,
}


def repair_and_trim(sub, region_levels):
    """補上最低等級與仍未覆蓋的 target，再把等級改到目標最小。

    修復：先把中繼 sensor 開到最低等級，再每次挑「新增覆蓋數 ÷ 目標增加量」
    最高的升級。修剪：逐顆嘗試改成仍可行、且目標最小的較低等級，直到
    不再改善。
    """
    levels = np.maximum(np.asarray(region_levels, dtype=int), sub.min_level)
    while True:
        uncovered = ~sub.covered(levels)
        if not uncovered.any():
            break
        current = sub.objective(levels)
        best = None
        for j in range(sub.size):
            for level in range(levels[j] + 1, int(sub.option_counts[j])):
                reach = (sub.lmin[:, j] >= 1) & (sub.lmin[:, j] <= level)
                gain = int(np.count_nonzero(reach & uncovered))
                if gain == 0:
                    continue
                trial = levels.copy()
                trial[j] = level
                delta = sub.objective(trial) - current
                score = gain / max(delta, 1e-12)
                if best is None or score > best[0]:
                    best = (score, j, level)
        if best is None:
            break  # 剩下的 target 區域內無法覆蓋
        levels[best[1]] = best[2]

    improved = True
    while improved:
        improved = False
        for j in range(sub.size):
            lower = np.arange(sub.min_level[j], levels[j])
            if len(lower) == 0:
                continue
            trials = np.repeat(levels[None, :], len(lower), axis=0)
            trials[:, j] = lower
            feasible = sub.feasible_batch(trials)
            if not feasible.any():
                continue
            values = sub.objective_batch(trials[feasible])
            position = int(np.argmin(values))
            if values[position] < sub.objective(levels) - 1e-12:
                levels = trials[feasible][position]
                improved = True
    return levels


__all__ = [
    "DEFAULT_FOCUS_POWER",
    "DEFAULT_PENALTY",
    "ENCODINGS",
    "LevelEncoding",
    "LevelSubproblem",
    "OBJECTIVES",
    "OneHotEncoding",
    "ThermometerEncoding",
    "apply_with_fixed_routes",
    "build_level_subproblem",
    "repair_and_trim",
    "slack_weights",
    "solve_exact",
]
