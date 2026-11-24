"""
Capa 1 — Estrategias Primarias Económicas (AFML Cap. 3.6).

Responsabilidad exclusiva:
- Proponer una dirección candidata causal `side ∈ {-1, 0, +1}` al cierre de cada barra t0
  basada en una hipótesis económica transparente (alto Recall, cero redes neuronales).
- NUNCA decide el tamaño de la apuesta ni filtra fricción: eso pertenece al Meta-Modelo (Capa 2)
  y a los Risk Overlays (Capa 3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import pandas as pd


class BasePrimaryStrategy(ABC):
    """Interfaz base para todo Modelo Primario generador de dirección."""

    name: str = "BasePrimaryStrategy"

    def __init__(self, bilateral: bool = True):
        self.bilateral = bool(bilateral)

    @abstractmethod
    def generate_sides(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula la serie causal de direcciones propuestas en {-1, 0, +1}.
        Debe usar exclusivamente información disponible hasta el cierre de t.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {"bilateral": self.bilateral}


class TrendPullbackPrimary(BasePrimaryStrategy):
    """
    Hipótesis Económica: Prima de Momentum Tendencial (TSMOM / Filtro Estructural SMA).
    - Long (+1) cuando Close > SMA(sma_slow) y EMA(ema_fast) > EMA(ema_slow).
    - Short (-1) cuando bilateral=True, Close < SMA(sma_slow) y EMA(ema_fast) < EMA(ema_slow).
    - Neutral (0) en zonas de transición o warmup.
    """

    name = "TrendPullbackPrimary"

    def __init__(
        self,
        sma_slow: int = 200,
        ema_fast: int = 20,
        ema_slow: int = 50,
        bilateral: bool = True,
    ):
        super().__init__(bilateral=bilateral)
        self.sma_slow = int(sma_slow)
        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)

    def generate_sides(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("Falta columna 'close' para TrendPullbackPrimary.")

        close = df["close"].astype(float)
        sma_slow = close.rolling(self.sma_slow, min_periods=self.sma_slow).mean()
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()

        sides = np.zeros(len(df), dtype=np.int8)
        valid = ~sma_slow.isna()

        long_cond = valid & (close > sma_slow) & (ema_fast > ema_slow)
        sides = np.where(long_cond, 1, sides)

        if self.bilateral:
            short_cond = valid & (close < sma_slow) & (ema_fast < ema_slow)
            sides = np.where(short_cond, -1, sides)

        return pd.Series(sides.astype(np.int8), index=df.index, name="primary_side")

    def get_params(self) -> Dict[str, Any]:
        return {
            "sma_slow": self.sma_slow,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "bilateral": self.bilateral,
        }


class DonchianBreakoutPrimary(BasePrimaryStrategy):
    """
    Hipótesis Económica: Ruptura de Canales de Liquidez (Donchian Channel).
    Usa estrictamente extremos de barras previas (shift(1)) para evitar lookahead intrabar.
    """

    name = "DonchianBreakoutPrimary"

    def __init__(self, lookback: int = 20, bilateral: bool = True):
        super().__init__(bilateral=bilateral)
        self.lookback = int(lookback)

    def generate_sides(self, df: pd.DataFrame) -> pd.Series:
        for col in ("high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Falta columna '{col}' para DonchianBreakoutPrimary.")

        close = df["close"].astype(float)
        upper = df["high"].astype(float).shift(1).rolling(self.lookback, min_periods=self.lookback).max()
        lower = df["low"].astype(float).shift(1).rolling(self.lookback, min_periods=self.lookback).min()

        sides = np.zeros(len(df), dtype=np.int8)
        sides = np.where((~upper.isna()) & (close > upper), 1, sides)
        if self.bilateral:
            sides = np.where((~lower.isna()) & (close < lower), -1, sides)

        return pd.Series(sides.astype(np.int8), index=df.index, name="primary_side")

    def get_params(self) -> Dict[str, Any]:
        return {"lookback": self.lookback, "bilateral": self.bilateral}


class VolatilityMeanReversionPrimary(BasePrimaryStrategy):
    """
    Hipótesis Económica: Reversión de Sobrerreacción de Corto Plazo alineada a tendencia macro.
    - Long (+1): Close > SMA(sma_slow) y z-score del retorno de z_window días < -z_threshold.
    - Short (-1): bilateral=True, Close < SMA(sma_slow) y z-score > +z_threshold.
    """

    name = "VolatilityMeanReversionPrimary"

    def __init__(
        self,
        sma_slow: int = 200,
        z_window: int = 5,
        vol_window: int = 20,
        z_threshold: float = 1.5,
        bilateral: bool = True,
    ):
        super().__init__(bilateral=bilateral)
        self.sma_slow = int(sma_slow)
        self.z_window = int(z_window)
        self.vol_window = int(vol_window)
        self.z_threshold = float(z_threshold)

    def generate_sides(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("Falta columna 'close' para VolatilityMeanReversionPrimary.")

        close = df["close"].astype(float)
        sma_slow = close.rolling(self.sma_slow, min_periods=self.sma_slow).mean()
        ret_n = np.log(close / close.shift(self.z_window))
        vol_daily = np.log(close / close.shift(1)).rolling(self.vol_window, min_periods=self.vol_window).std()
        z_score = ret_n / (vol_daily * np.sqrt(self.z_window) + 1e-12)

        sides = np.zeros(len(df), dtype=np.int8)
        valid = (~sma_slow.isna()) & (~z_score.isna())

        sides = np.where(valid & (close > sma_slow) & (z_score < -self.z_threshold), 1, sides)
        if self.bilateral:
            sides = np.where(valid & (close < sma_slow) & (z_score > self.z_threshold), -1, sides)

        return pd.Series(sides.astype(np.int8), index=df.index, name="primary_side")

    def get_params(self) -> Dict[str, Any]:
        return {
            "sma_slow": self.sma_slow,
            "z_window": self.z_window,
            "vol_window": self.vol_window,
            "z_threshold": self.z_threshold,
            "bilateral": self.bilateral,
        }


_STRATEGY_REGISTRY = {
    "TrendPullbackPrimary": TrendPullbackPrimary,
    "DonchianBreakoutPrimary": DonchianBreakoutPrimary,
    "VolatilityMeanReversionPrimary": VolatilityMeanReversionPrimary,
}


def get_primary_strategy(name: str, **params) -> BasePrimaryStrategy:
    """Instancia una estrategia primaria desde su nombre y parámetros (para ModelBundle v3.0)."""
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"Estrategia primaria desconocida: '{name}'. Disponibles: {list(_STRATEGY_REGISTRY)}")
    return _STRATEGY_REGISTRY[name](**params)
