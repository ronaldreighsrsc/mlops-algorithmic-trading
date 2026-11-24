"""
Validación Cruzada Purgada con Embargo basada en intervalos exactos [t0, t1] (AFML Cap. 7).

Reglas innegociables de integridad temporal:
1. Purging: cualquier muestra de entrenamiento i cuyo intervalo [t0_i, t1_i] se solape
   con el intervalo de validación [T_val_start, T_val_end] se elimina físicamente.
2. Embargo: cualquier muestra de entrenamiento i cuyo t0_i caiga en
   (T_val_end, T_val_end + embargo] se elimina para neutralizar correlación serial residual.
3. Walk-Forward: en el instante T_now solo se permiten eventos estrictamente resueltos
   (t1 < T_now).
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """
    Implementación fiel de Purged K-Fold CV (Marcos López de Prado, AFML Cap. 7.4).
    Acepta marcas temporales exactas (t0 = inicio del evento, t1 = cierre de barrera).
    """

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits debe ser >= 2.")
        if not (0.0 <= pct_embargo < 0.5):
            raise ValueError("pct_embargo debe estar en [0.0, 0.5).")
        self.n_splits = int(n_splits)
        self.pct_embargo = float(pct_embargo)

    def split(
        self,
        t0: pd.Series | pd.Index | np.ndarray,
        t1: pd.Series | pd.Index | np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera pares (train_indices, val_indices) garantizando cero solapamiento
        entre [t0_train, t1_train] y [min(t0_val), max(t1_val)].
        """
        t0_arr = np.asarray(t0)
        t1_arr = np.asarray(t1)
        n = len(t0_arr)
        if n != len(t1_arr):
            raise ValueError("t0 y t1 deben tener exactamente la misma longitud.")
        if n < self.n_splits:
            raise ValueError(f"Insuficientes muestras ({n}) para {self.n_splits} splits.")

        indices = np.arange(n)
        embargo_count = int(np.ceil(n * self.pct_embargo))

        # Particiones contiguas cronológicas
        test_ranges = [
            (arr[0], arr[-1] + 1)
            for arr in np.array_split(indices, self.n_splits)
            if len(arr) > 0
        ]

        for val_start_idx, val_end_idx in test_ranges:
            val_indices = indices[val_start_idx:val_end_idx]

            # Fronteras temporales reales del fold de validación
            val_t0_min = t0_arr[val_indices[0]]
            val_t1_max = np.max(t1_arr[val_indices])

            # Límite de embargo posterior al fold de validación
            embargo_end_pos = min(n - 1, val_end_idx - 1 + embargo_count)
            embargo_t0_limit = t0_arr[embargo_end_pos]

            train_mask = np.ones(n, dtype=bool)
            train_mask[val_indices] = False

            # 1. Purging: eliminar todo evento j tal que [t0_j, t1_j] intersecta [val_t0_min, val_t1_max]
            overlaps = (t1_arr >= val_t0_min) & (t0_arr <= val_t1_max)
            train_mask &= ~overlaps

            # 2. Embargo: eliminar eventos inmediatamente posteriores al bloque de prueba
            if embargo_count > 0 and val_end_idx < n:
                in_embargo = (t0_arr > val_t0_min) & (t0_arr <= embargo_t0_limit)
                train_mask &= ~in_embargo

            train_indices = indices[train_mask]
            yield train_indices, val_indices


def get_causal_walk_forward_train_mask(
    t1_series: pd.Series | np.ndarray,
    current_time: object,
) -> np.ndarray:
    """
    Retorna máscara booleana estrictamente causal para reentrenamiento Walk-Forward:
    solo eventos cuya barrera t1 ya cerró ESTRICTAMENTE ANTES de current_time.
    """
    t1_arr = np.asarray(t1_series)
    return t1_arr < current_time
