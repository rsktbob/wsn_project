"""QUBO 的經典求解器：模擬退火（SA）與小題目用的窮舉。"""

from __future__ import annotations

import numpy as np

from Problem.services.evaluation_kernels import njit


@njit(cache=True, nogil=True)
def _anneal_kernel(h, coupling, x, flips, t_start, t_end, choices, draws):
    """單一 read 的 SA；每步隨機翻一個 bit，溫度等比下降。

    ``field[j] = h_j + Σ_k J_jk x_k``，翻轉 x_j 的能量變化為
    ``(1 − 2x_j) · field[j]``；接受後只需以 J 的第 j 欄更新 field。
    """
    n = h.shape[0]
    field = h.copy()
    for j in range(n):
        if x[j] == 1:
            for k in range(n):
                field[k] += coupling[k, j]
    energy_change = 0.0
    best_change = 0.0
    best_x = x.copy()
    ratio = t_end / t_start
    for step in range(flips):
        temperature = t_start * ratio ** (step / max(1, flips - 1))
        j = choices[step]
        delta = (1 - 2 * x[j]) * field[j]
        if delta <= 0.0 or draws[step] < np.exp(-delta / temperature):
            sign = 1.0 if x[j] == 0 else -1.0
            x[j] = 1 - x[j]
            for k in range(n):
                field[k] += sign * coupling[k, j]
            energy_change += delta
            if energy_change < best_change:
                best_change = energy_change
                best_x[:] = x
    return best_x


def simulated_annealing(
    model,
    sweeps=200,
    num_reads=10,
    t_start=2.0,
    t_end=0.01,
    rng=None,
):
    """回傳 ``(最佳 x, 最佳能量, 各 read 的最終能量)``。

    每個 read 從隨機 0/1 出發，共翻 ``sweeps × 變數數`` 次；溫度單位與
    縮放後的負擔、懲罰相同（預設懲罰為 2）。
    """
    if rng is None:
        rng = np.random.default_rng()
    h, coupling, offset = model.to_arrays()
    n = len(h)
    flips = int(sweeps) * n
    best_x = None
    best_energy = np.inf
    read_energies = []
    for _ in range(int(num_reads)):
        x = rng.integers(0, 2, n).astype(np.int64)
        choices = rng.integers(0, n, flips).astype(np.int64)
        draws = rng.random(flips)
        found = _anneal_kernel(
            h, coupling, x, flips, float(t_start), float(t_end), choices, draws
        )
        energy = float(offset + h @ found + 0.5 * found @ coupling @ found)
        read_energies.append(energy)
        if energy < best_energy:
            best_energy = energy
            best_x = found.copy()
    return best_x, best_energy, np.asarray(read_energies)


def brute_force(model, max_variables=22, chunk=1 << 16):
    """列舉全部 2ⁿ 個組合，回傳 ``(最佳 x, 最佳能量)``；只給小題目驗證用。"""
    h, coupling, offset = model.to_arrays()
    n = len(h)
    if n > max_variables:
        raise ValueError(f"{n} variables is too many for brute force")
    best_x = None
    best_energy = np.inf
    shifts = np.arange(n - 1, -1, -1)
    for start in range(0, 1 << n, chunk):
        indices = np.arange(start, min(start + chunk, 1 << n))
        bits = ((indices[:, None] >> shifts) & 1).astype(float)
        energies = offset + bits @ h + 0.5 * np.einsum(
            "ij,jk,ik->i", bits, coupling, bits
        )
        position = int(np.argmin(energies))
        if energies[position] < best_energy:
            best_energy = float(energies[position])
            best_x = bits[position].astype(np.int64)
    return best_x, best_energy


__all__ = ["brute_force", "simulated_annealing"]
