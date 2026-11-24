"""
Pesos por Unicidad de Muestra, Atribución por Retorno y Time Decay (AFML Cap. 4).

Corrige la violación del supuesto IID causada por eventos Triple Barrier solapados
en el tiempo:
1. Concurrencia por barra: c_t = sum_i I(t0_i <= t <= t1_i)
2. Unicidad media del evento i: u_bar_i = mean_{t in [t0_i, t1_i]}(1 / c_t)
3. Atribución por retorno absoluto: w_tilde_i = |sum_{t in [t0_i, t1_i]} r_t / c_t|
4. Decaimiento temporal lineal sobre unicidad acumulada (AFML §4.7).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_concurrency(
    t0_pos: np.ndarray,
    t1_pos: np.ndarray,
    n_bars: int,
) -> np.ndarray:
    """Cuenta cuántos eventos están simultáneamente activos en cada barra t ∈ [0, n_bars)."""
    c = np.zeros(int(n_bars), dtype=np.float64)
    t0_arr = np.asarray(t0_pos, dtype=np.int64)
    t1_arr = np.asarray(t1_pos, dtype=np.int64)

    for s, e in zip(t0_arr, t1_arr):
        s_clip = max(0, int(s))
        e_clip = min(int(n_bars) - 1, int(e))
        if s_clip <= e_clip:
            c[s_clip : e_clip + 1] += 1.0
    return c


def compute_average_uniqueness(
    t0_pos: np.ndarray,
    t1_pos: np.ndarray,
    n_bars: int,
) -> np.ndarray:
    """Calcula la unicidad promedio u_bar_i ∈ (0, 1] de cada evento."""
    t0_arr = np.asarray(t0_pos, dtype=np.int64)
    t1_arr = np.asarray(t1_pos, dtype=np.int64)
    n_events = len(t0_arr)
    if n_events == 0:
        return np.array([], dtype=np.float64)

    c = compute_concurrency(t0_arr, t1_arr, n_bars)
    avg_u = np.zeros(n_events, dtype=np.float64)

    for idx, (s, e) in enumerate(zip(t0_arr, t1_arr)):
        s_clip = max(0, int(s))
        e_clip = min(int(n_bars) - 1, int(e))
        if s_clip <= e_clip:
            inv_c = 1.0 / np.maximum(c[s_clip : e_clip + 1], 1.0)
            avg_u[idx] = float(np.mean(inv_c))
        else:
            avg_u[idx] = 1.0
    return avg_u


def apply_time_decay(
    avg_uniqueness: np.ndarray,
    decay_c: float = 0.5,
) -> np.ndarray:
    """
    Aplica decaimiento lineal por unicidad acumulada (AFML §4.7):
    - decay_c = 1.0: sin decaimiento (todos los eventos igual de vigentes).
    - 0 <= decay_c < 1: el evento más antiguo recibe factor `decay_c` y el más reciente 1.0.
    - -1 < decay_c < 0: los eventos más antiguos se podan a peso 0.
    """
    u = np.asarray(avg_uniqueness, dtype=np.float64)
    if len(u) == 0:
        return np.array([], dtype=np.float64)
    if decay_c == 1.0 or len(u) == 1:
        return np.ones_like(u, dtype=np.float64)

    cum_u = np.cumsum(u)
    total_u = float(cum_u[-1])
    if total_u <= 0.0:
        return np.ones_like(u, dtype=np.float64)

    if decay_c >= 0.0:
        slope = (1.0 - decay_c) / total_u
    else:
        slope = 1.0 / ((decay_c + 1.0) * total_u)
    const = 1.0 - slope * total_u
    weights = const + slope * cum_u
    return np.maximum(weights, 0.0)


def compute_sample_weights(
    t0_pos: np.ndarray,
    t1_pos: np.ndarray,
    n_bars: int,
    close_prices: Optional[np.ndarray] = None,
    decay_c: float = 0.5,
    use_return_attribution: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula pesos finales de muestra w_i normalizados tal que sum(w_i) == N_events.
    Retorna: (sample_weights, avg_uniqueness).
    """
    t0_arr = np.asarray(t0_pos, dtype=np.int64)
    t1_arr = np.asarray(t1_pos, dtype=np.int64)
    n_events = len(t0_arr)
    if n_events == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    c = compute_concurrency(t0_arr, t1_arr, n_bars)
    avg_u = compute_average_uniqueness(t0_arr, t1_arr, n_bars)

    if use_return_attribution and close_prices is not None:
        prices = np.asarray(close_prices, dtype=np.float64)
        log_rets = np.zeros(len(prices), dtype=np.float64)
        valid = (prices[1:] > 0) & (prices[:-1] > 0)
        log_rets[1:] = np.where(valid, np.log(prices[1:] / prices[:-1]), 0.0)

        raw_w = np.zeros(n_events, dtype=np.float64)
        for idx, (s, e) in enumerate(zip(t0_arr, t1_arr)):
            s_clip = max(1, int(s) + 1)
            e_clip = min(len(log_rets) - 1, int(e))
            if s_clip <= e_clip:
                raw_w[idx] = abs(float(np.sum(log_rets[s_clip : e_clip + 1] / np.maximum(c[s_clip : e_clip + 1], 1.0))))
            else:
                raw_w[idx] = avg_u[idx]
        # Respaldo suave si alguna muestra tuvo retorno atribuido 0
        eps_floor = float(np.mean(raw_w[raw_w > 0])) * 0.05 if np.any(raw_w > 0) else 1.0
        raw_w = np.where(raw_w > 0, raw_w, eps_floor)
    else:
        raw_w = avg_u.copy()

    decay_factors = apply_time_decay(avg_u, decay_c=decay_c)
    combined = raw_w * decay_factors

    total = float(np.sum(combined))
    if total <= 0.0:
        weights = np.ones(n_events, dtype=np.float64)
    else:
        weights = combined * (n_events / total)

    return weights, avg_u
