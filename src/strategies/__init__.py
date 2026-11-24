"""Capa 1 — Estrategias Primarias Económicas (Generadores de Lado side ∈ {-1, 0, +1})."""

from src.strategies.primary_strategies import (
    BasePrimaryStrategy,
    TrendPullbackPrimary,
    DonchianBreakoutPrimary,
    VolatilityMeanReversionPrimary,
    get_primary_strategy,
)

__all__ = [
    "BasePrimaryStrategy",
    "TrendPullbackPrimary",
    "DonchianBreakoutPrimary",
    "VolatilityMeanReversionPrimary",
    "get_primary_strategy",
]
