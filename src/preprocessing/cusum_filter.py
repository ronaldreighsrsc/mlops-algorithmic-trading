"""
Filtro CUSUM Simétrico Dinámico para Muestreo de Eventos de Información (AFML §2.5.2.1).

Evita entrenar modelos sobre barras muertas o ruido lateral disparando eventos t0
únicamente cuando el desplazamiento acumulado de retornos logarítmicos excede un
umbral h_t calibrado causalmente por la volatilidad condicional del día anterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CUSUMResult:
    event_positions: np.ndarray
    event_directions: np.ndarray
    bars_since_last: np.ndarray


class SymmetricCUSUMFilter:
    """
    Filtro CUSUM Simétrico (Marcos López de Prado, AFML §2.5.2.1):
        S_t^+ = max(0, S_{t-1}^+ + r_t)
        S_t^- = min(0, S_{t-1}^- + r_t)
    Dispara evento en t si S_t^+ >= h_t (resetea S^+) o S_t^- <= -h_t (resetea S^-).
    """

    def __init__(
        self,
        kappa: float = 1.0,
        vol_col: Optional[str] = "EGARCH_Vol",
        vol_is_pct: bool = True,
        static_threshold: Optional[float] = None,
        min_threshold: float = 1e-4,
    ):
        if kappa <= 0.0:
            raise ValueError("kappa debe ser > 0.")
        self.kappa = float(kappa)
        self.vol_col = vol_col
        self.vol_is_pct = bool(vol_is_pct)
        self.static_threshold = static_threshold
        self.min_threshold = float(min_threshold)

    def filter_events(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> CUSUMResult:
        """
        Retorna las posiciones enteras de eventos disparados de forma 100% causal.
        Usa la volatilidad del cierre t-1 (shift(1)) para fijar h_t sin mirar el futuro.
        """
        if price_col not in df.columns:
            raise ValueError(f"Falta columna '{price_col}' para el Filtro CUSUM.")

        closes = df[price_col].astype(float).values
        n = len(closes)
        if n < 3:
            return CUSUMResult(
                event_positions=np.array([], dtype=np.int64),
                event_directions=np.array([], dtype=np.int8),
                bars_since_last=np.array([], dtype=np.int64),
            )

        # Retornos logarítmicos estrictamente causales
        safe_closes = np.where(closes > 0, closes, np.nan)
        log_rets = np.zeros(n, dtype=np.float64)
        valid_ratio = (safe_closes[1:] > 0) & (safe_closes[:-1] > 0)
        log_rets[1:] = np.where(valid_ratio, np.log(safe_closes[1:] / safe_closes[:-1]), 0.0)

        # Umbral dinámico causal h_t basado en sigma_{t-1}
        if self.static_threshold is not None:
            thresholds = np.full(n, max(self.min_threshold, float(self.static_threshold)), dtype=np.float64)
        elif self.vol_col and self.vol_col in df.columns:
            vol_series = df[self.vol_col].astype(float).shift(1).ffill()
            vol_vals = vol_series.values / 100.0 if self.vol_is_pct else vol_series.values
            fallback_vol = float(np.nanstd(log_rets[: min(n, 60)])) or 0.01
            vol_vals = np.where(np.isnan(vol_vals) | (vol_vals <= 0), fallback_vol, vol_vals)
            thresholds = np.maximum(self.min_threshold, self.kappa * vol_vals)
        else:
            # Volatilidad rodante causal de 20 barras desplazada 1 paso
            rolling_vol = pd.Series(log_rets).rolling(20, min_periods=5).std().shift(1).ffill().fillna(0.01).values
            thresholds = np.maximum(self.min_threshold, self.kappa * rolling_vol)

        s_pos = 0.0
        s_neg = 0.0
        last_event_idx = 0

        event_positions = []
        event_directions = []
        bars_since_last = []

        for t in range(1, n):
            r_t = log_rets[t]
            h_t = thresholds[t]

            s_pos = max(0.0, s_pos + r_t)
            s_neg = min(0.0, s_neg + r_t)

            if s_pos >= h_t:
                s_pos = 0.0
                event_positions.append(t)
                event_directions.append(1)
                bars_since_last.append(t - last_event_idx)
                last_event_idx = t
            elif s_neg <= -h_t:
                s_neg = 0.0
                event_positions.append(t)
                event_directions.append(-1)
                bars_since_last.append(t - last_event_idx)
                last_event_idx = t

        return CUSUMResult(
            event_positions=np.asarray(event_positions, dtype=np.int64),
            event_directions=np.asarray(event_directions, dtype=np.int8),
            bars_since_last=np.asarray(bars_since_last, dtype=np.int64),
        )
