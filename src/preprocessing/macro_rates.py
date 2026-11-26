"""
Slice A (Quant v3.1): Extractor Causal de Diferenciales de Tasas Reales,
Estructura Temporal de Volatilidad (VIX/VIX3M), Crédito y Ratio Oro/Plata.

Reglas estrictas Anti-Leakage:
1. Toda serie macro de cierre diario se desplaza `shift(1)` (dato conocido al cierre de T-1)
   antes de alinearse con la barra T del activo objetivo mediante unión causal backward.
2. Cero `.bfill()` en toda la tubería (nunca propagar valores futuros hacia el pasado).
3. Cache local determinista en `data/raw/macro/` para reproducibilidad offline y rapidez.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MacroRatesProvider")

FRED_DAILY_SERIES: Dict[str, str] = {
    "US2Y": "DGS2",
    "US10Y": "DGS10",
    "US_REAL_10Y": "DFII10",
    "US_BREAKEVEN_10Y": "T10YIE",
    "US_CURVE_10Y2Y": "T10Y2Y",
    "ECB_DFR": "ECBDFR",
    "FED_FUNDS": "DFF",
}

YF_DAILY_SERIES: Dict[str, str] = {
    "VIX3M": "^VIX3M",
    "VIX_YF": "^VIX",
    "HYG": "HYG",
    "IEF": "IEF",
}


class MacroRatesProvider:
    """Descarga, cachea y alinea causalmente series macro diarias de FRED, yfinance y MT5 local."""

    def __init__(self, base_dir: Optional[str] = None, cache_ttl_days: float = 1.0):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "data", "raw")
        self.cache_dir = os.path.join(self.raw_dir, "macro")
        self.cache_ttl_days = float(cache_ttl_days)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _is_cache_fresh(self, path: str, force_refresh: bool) -> bool:
        if force_refresh or not os.path.exists(path):
            return False
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        return age_days <= self.cache_ttl_days

    def fetch_fred_series(self, series_id: str, force_refresh: bool = False) -> pd.Series:
        """Descarga serie diaria pública de FRED con caché CSV local (sin API key)."""
        cache_path = os.path.join(self.cache_dir, f"fred_{series_id}.csv")
        if self._is_cache_fresh(cache_path, force_refresh):
            try:
                df_c = pd.read_csv(cache_path)
                df_c["date"] = pd.to_datetime(df_c["date"]).dt.normalize()
                return df_c.set_index("date")["value"].astype(float).sort_index()
            except Exception:
                pass

        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            df = pd.read_csv(url, timeout=15) if "timeout" in pd.read_csv.__code__.co_varnames else pd.read_csv(url)
            date_col, val_col = df.columns[0], df.columns[1]
            df = df.rename(columns={date_col: "date", val_col: "value"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"]).sort_values("date")
            df.to_csv(cache_path, index=False)
            return df.set_index("date")["value"].astype(float)
        except Exception as exc:
            logger.warning(f"No se pudo descargar FRED {series_id}: {exc}")
            if os.path.exists(cache_path):
                df_c = pd.read_csv(cache_path)
                df_c["date"] = pd.to_datetime(df_c["date"]).dt.normalize()
                return df_c.set_index("date")["value"].astype(float).sort_index()
            return pd.Series(dtype=float)

    def fetch_yfinance_series(self, ticker: str, alias: str, force_refresh: bool = False) -> pd.Series:
        """Descarga serie diaria de cierre vía yfinance con caché local."""
        safe_name = alias.lower()
        cache_path = os.path.join(self.cache_dir, f"yf_{safe_name}.csv")
        if self._is_cache_fresh(cache_path, force_refresh):
            try:
                df_c = pd.read_csv(cache_path)
                df_c["date"] = pd.to_datetime(df_c["date"]).dt.normalize()
                return df_c.set_index("date")["value"].astype(float).sort_index()
            except Exception:
                pass

        try:
            import yfinance as yf

            df = yf.download(ticker, start="2000-01-01", progress=False)
            if not df.empty:
                close_col = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
                out = pd.DataFrame({
                    "date": pd.to_datetime(close_col.index).tz_localize(None).normalize(),
                    "value": pd.to_numeric(close_col.values, errors="coerce"),
                }).dropna().drop_duplicates(subset=["date"]).sort_values("date")
                out.to_csv(cache_path, index=False)
                return out.set_index("date")["value"].astype(float)
        except Exception as exc:
            logger.warning(f"No se pudo descargar yfinance {ticker}: {exc}")

        if os.path.exists(cache_path):
            df_c = pd.read_csv(cache_path)
            df_c["date"] = pd.to_datetime(df_c["date"]).dt.normalize()
            return df_c.set_index("date")["value"].astype(float).sort_index()
        return pd.Series(dtype=float)

    def load_local_silver_series(self) -> pd.Series:
        """Carga cierre histórico de Plata (XAGUSD) desde data/raw/ local si existe."""
        xag_path = os.path.join(self.raw_dir, "XAGUSD_daily.csv")
        if not os.path.exists(xag_path):
            return pd.Series(dtype=float)
        try:
            df = pd.read_csv(xag_path)
            df["date"] = pd.to_datetime(df["time"]).dt.normalize()
            df = df.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"]).sort_values("date")
            return df.set_index("date")["close"].astype(float)
        except Exception:
            return pd.Series(dtype=float)

    def fetch_all_macro_frame(self, force_refresh: bool = False) -> pd.DataFrame:
        """Consolida todas las series macro diarias crudas (sin alinear ni shiftear aún)."""
        cols: Dict[str, pd.Series] = {}
        for col_name, fred_id in FRED_DAILY_SERIES.items():
            s = self.fetch_fred_series(fred_id, force_refresh=force_refresh)
            if not s.empty:
                cols[col_name] = s

        for alias, yf_ticker in YF_DAILY_SERIES.items():
            s = self.fetch_yfinance_series(yf_ticker, alias=alias, force_refresh=force_refresh)
            if not s.empty:
                cols[alias] = s

        xag_s = self.load_local_silver_series()
        if not xag_s.empty:
            cols["XAG_CLOSE"] = xag_s

        if not cols:
            return pd.DataFrame()

        df_macro = pd.DataFrame(cols).sort_index()
        df_macro.index.name = "date"
        # Forward-fill solo por feriados locales; NUNCA .bfill()
        df_macro = df_macro.ffill()
        return df_macro

    @staticmethod
    def align_causal_macro(
        df_asset: pd.DataFrame,
        df_macro_raw: pd.DataFrame,
        time_col: str = "time",
    ) -> pd.DataFrame:
        """
        Alinea causalmente el panel macro diario a las fechas de `df_asset` aplicando
        ESTRICTAMENTE información cerrada del día anterior (< fecha de barra T, `shift(1)`).
        Garantiza cero fuga futura y cero `.bfill()`.
        """
        out = df_asset.copy()
        if df_macro_raw.empty:
            return out

        if time_col in out.columns:
            asset_dates = pd.to_datetime(out[time_col]).dt.tz_localize(None)
        elif isinstance(out.index, pd.DatetimeIndex):
            asset_dates = pd.to_datetime(out.index).tz_localize(None)
        else:
            raise ValueError(f"Se requiere columna '{time_col}' o DatetimeIndex para alinear macro causalmente.")

        macro_sorted = df_macro_raw.copy().sort_index()
        macro_sorted.index = pd.to_datetime(macro_sorted.index).tz_localize(None).normalize()
        # Regla estricta: en fecha T solo es visible la observación más reciente con date < T
        # `merge_asof(..., allow_exact_matches=False)` implementa matemáticamente T_macro < T_asset
        left_df = pd.DataFrame({"_asset_dt": asset_dates.dt.normalize(), "_orig_pos": np.arange(len(out))})
        left_sorted = left_df.sort_values("_asset_dt")

        right_df = macro_sorted.reset_index().rename(columns={macro_sorted.index.name or "index": "_macro_dt"})
        merged = pd.merge_asof(
            left_sorted,
            right_df,
            left_on="_asset_dt",
            right_on="_macro_dt",
            direction="backward",
            allow_exact_matches=False,
        )
        merged = merged.sort_values("_orig_pos").reset_index(drop=True)

        # Inyectar columnas derivadas causales (diferenciales y pendientes)
        if "US2Y" in merged.columns and "ECB_DFR" in merged.columns:
            out["rate_diff_us2y_ecb"] = merged["US2Y"].values - merged["ECB_DFR"].values
            out["delta_rate_diff_5d"] = pd.Series(out["rate_diff_us2y_ecb"]).diff(5).values
            out["delta_rate_diff_20d"] = pd.Series(out["rate_diff_us2y_ecb"]).diff(20).values

        if "FED_FUNDS" in merged.columns and "ECB_DFR" in merged.columns:
            out["rate_diff_fed_ecb"] = merged["FED_FUNDS"].values - merged["ECB_DFR"].values

        if "US_CURVE_10Y2Y" in merged.columns:
            out["yield_curve_10y_2y"] = merged["US_CURVE_10Y2Y"].values
        elif "US10Y" in merged.columns and "US2Y" in merged.columns:
            out["yield_curve_10y_2y"] = merged["US10Y"].values - merged["US2Y"].values

        if "US_REAL_10Y" in merged.columns:
            out["us_real_yield_10y"] = merged["US_REAL_10Y"].values
            out["delta_real_yield_5d"] = pd.Series(out["us_real_yield_10y"]).diff(5).values
            out["delta_real_yield_20d"] = pd.Series(out["us_real_yield_10y"]).diff(20).values

        if "US_BREAKEVEN_10Y" in merged.columns:
            out["breakeven_infl_10y"] = merged["US_BREAKEVEN_10Y"].values
            out["delta_breakeven_20d"] = pd.Series(out["breakeven_infl_10y"]).diff(20).values

        if "VIX3M" in merged.columns:
            vix3m = merged["VIX3M"].astype(float).values
            if "VIX_close" in out.columns:
                # Usar VIX_close del activo si ya viene alineado, o VIX_YF shifteado si no
                vix_base = out["VIX_close"].astype(float).values
            elif "VIX_YF" in merged.columns:
                vix_base = merged["VIX_YF"].astype(float).values
            else:
                vix_base = np.full(len(out), np.nan)
            out["vix3m_close"] = vix3m
            out["vix_term_ratio"] = np.where(vix3m > 0, vix_base / vix3m, np.nan)
            out["vix_contango"] = vix3m - vix_base

        if "HYG" in merged.columns and "IEF" in merged.columns:
            hyg = merged["HYG"].astype(float).values
            ief = merged["IEF"].astype(float).values
            ratio = np.where(ief > 0, hyg / ief, np.nan)
            out["credit_hy_ief_ratio"] = ratio
            out["delta_credit_ratio_20d"] = pd.Series(ratio).pct_change(20).values

        if "XAG_CLOSE" in merged.columns and "close" in out.columns:
            xag = merged["XAG_CLOSE"].astype(float).values
            # Comparar cierre previo de ambos metales para evitar lookahead intrabar si se usa en features
            close_prev = out["close"].astype(float).shift(1).values
            gsr = np.where(xag > 0, close_prev / xag, np.nan)
            out["gold_silver_ratio"] = gsr
            out["delta_gsr_20d"] = pd.Series(gsr).pct_change(20).values

        return out

    def enrich_dataframe(self, df_asset: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
        """Descarga/carga caché macro y acopla las variables derivadas causalmente."""
        df_macro = self.fetch_all_macro_frame(force_refresh=force_refresh)
        return self.align_causal_macro(df_asset, df_macro)
