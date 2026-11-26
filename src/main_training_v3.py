"""
Pipeline Maestro de Entrenamiento Meta-Labeling Institucional v3.0 / v3.1.

Modos soportados (SPEC_QUANT_V3_1_ALPHA_SIGNALS):
  - d1_ohlcv:     Línea base v3.0 con TrendPullbackPrimary sobre OHLCV/macro básico.
  - d1_macro:     Protocolo experimental cerrado v3.1 (N=6: 3 Primarias x 2 Modelos Tabulares)
                  enriquecido causalmente con diferenciales de tasas, TIPS DFII10, VIX/VIX3M,
                  crédito HYG/IEF, ratio Oro/Plata y Turn-of-the-Month.
  - intraday_tib: Pipeline intradía sobre Tick Imbalance Bars + VPIN desde ticks MT5 Parquet.

Ejemplos:
    python src/main_training_v3.py --mode d1_macro
    python src/main_training_v3.py --compare
    python src/main_training_v3.py --mode intraday_tib --asset EURUSD
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from typing import List

import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.models.meta_trainer_v3 import MetaModelTrainerV3, run_preregistered_v31_tournament
from src.preprocessing.macro_rates import MacroRatesProvider
from src.preprocessing.technical_features import TechnicalFeatureEngineer
from src.preprocessing.volatility import VolatilityModeler
from src.preprocessing.information_bars import (
    build_tick_imbalance_bars_with_microstructure,
    compute_tick_microstructure_features,
    filter_liquid_session_ticks,
)
from src.strategies.primary_strategies import BasePrimaryStrategy, get_primary_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MainTrainingV3")


def _get_preregistered_primaries_for_asset(activo: str) -> List[BasePrimaryStrategy]:
    act_up = activo.upper()
    is_fx = any(k in act_up for k in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "USD")) and "SP500" not in act_up and "XAU" not in act_up
    is_gold = ("ORO" in act_up) or ("XAU" in act_up)

    if is_fx:
        return [
            get_primary_strategy("TrendPullbackPrimary", sma_slow=200, ema_fast=20, ema_slow=50, bilateral=True),
            get_primary_strategy("DonchianBreakoutPrimary", lookback=20, bilateral=True),
            get_primary_strategy("MacroCarryMomentumPrimary", mode="fx_rate_diff", ema_window=50, bilateral=True),
        ]
    if is_gold:
        return [
            get_primary_strategy("TrendPullbackPrimary", sma_slow=200, ema_fast=20, ema_slow=50, bilateral=False),
            get_primary_strategy("DonchianBreakoutPrimary", lookback=20, bilateral=False),
            get_primary_strategy("MacroCarryMomentumPrimary", mode="real_yield", sma_window=100, bilateral=False),
        ]
    # Renta variable (SP500 / US500 / Índices)
    return [
        get_primary_strategy("TrendPullbackPrimary", sma_slow=200, ema_fast=20, ema_slow=50, bilateral=False),
        get_primary_strategy("DonchianBreakoutPrimary", lookback=20, bilateral=False),
        get_primary_strategy("VixRegimeEquityPrimary", vix_ratio_max=0.98, credit_delta_min=-0.03, sma_window=150, bilateral=False),
    ]


def _load_processed_asset(activo: str) -> pd.DataFrame | None:
    proc_path = os.path.join(base_dir, "data", "processed", f"{activo}_processed.csv")
    if not os.path.exists(proc_path):
        logger.warning(f"[{activo}] Dataset procesado no encontrado en {proc_path}. Omitido.")
        return None
    df = pd.read_csv(proc_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
    return df


def run_v3_training(
    assets: list[str],
    mode: str = "d1_macro",
    cusum_kappa: float = 1.0,
    enable_mda: bool = True,
    force_macro_refresh: bool = False,
) -> dict:
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_report = {}

    macro_df = pd.DataFrame()
    if mode == "d1_macro":
        logger.info("📡 [v3.1 Slice A] Cargando/actualizando series macro causales (FRED + yfinance + XAG local)...")
        provider = MacroRatesProvider(base_dir=base_dir, cache_ttl_days=1.0)
        macro_df = provider.fetch_all_macro_frame(force_refresh=force_macro_refresh)

    for activo in assets:
        bilateral = any(activo.upper().startswith(p) for p in ("EUR", "GBP", "USD")) and "SP500" not in activo.upper()
        k_up = 2.5 if ("ORO" in activo.upper() or "XAU" in activo.upper()) else 2.0

        if mode == "intraday_tib":
            ticks_path = os.path.join(base_dir, "data", "raw", "ticks", f"{activo}_ticks.parquet")
            if not os.path.exists(ticks_path):
                logger.warning(
                    f"[{activo}] No existe {ticks_path}. Ejecute primero extract_ticks_chunked() con MT5 conectado."
                )
                continue
            df_ticks = pd.read_parquet(ticks_path)
            df_ticks = filter_liquid_session_ticks(df_ticks, start_hour_utc=7, end_hour_utc=17)
            df_tib = build_tick_imbalance_bars_with_microstructure(df_ticks, expected_init_t=200, min_ticks=20, max_ticks=1500)
            df_tib = compute_tick_microstructure_features(df_tib)
            df_tib = TechnicalFeatureEngineer(target_price_col="close").add_indicators(df_tib)
            df_tib = VolatilityModeler(window_size=min(250, max(60, len(df_tib) // 5)), target_col="close").compute_egarch(
                df_tib, symbol=f"{activo}_tib"
            )
            primaries = _get_preregistered_primaries_for_asset(activo)[:2]
            res = run_preregistered_v31_tournament(
                symbol=f"{activo}_TIB",
                df=df_tib,
                primaries=primaries,
                results_dir=results_dir,
                cusum_kappa=cusum_kappa,
                k_up=k_up,
                k_down=1.5,
                max_hold=12,
                enable_mda_pruning=enable_mda,
                export_onnx=True,
            )
            summary_report[activo] = res["model_bundle"]["meta_model"]
            continue

        df = _load_processed_asset(activo)
        if df is None:
            continue

        if mode == "d1_ohlcv":
            primary = get_primary_strategy("TrendPullbackPrimary", sma_slow=200, ema_fast=20, ema_slow=50, bilateral=bilateral)
            logger.info(f"🚀 [v3.0 Baseline] Entrenando {activo} (d1_ohlcv, bilateral={bilateral})...")
            trainer = MetaModelTrainerV3(
                symbol=activo,
                primary_strategy=primary,
                cusum_kappa=cusum_kappa,
                k_up=k_up,
                k_down=1.5,
                max_hold=10,
                enable_mda_pruning=enable_mda,
            )
            res = trainer.train_and_audit(df=df, results_dir=results_dir, export_onnx=True)
            meta = res["model_bundle"]["meta_model"]
            summary_report[activo] = meta
            logger.info(
                f"✅ [{activo} d1_ohlcv] Ganador: {res['winner']} | CV LogLoss={meta['cv_log_loss']:.4f} | "
                f"OOS LogLoss={meta['oos_log_loss']:.4f} | ECE={meta['ece_oos']:.4f} | "
                f"DSR={meta['dsr_oos']:.3f} | CSCV PBO={meta['pbo_cscv']:.2%} | ONNX max_diff={meta['onnx_parity_max_diff']:.2e}"
            )
        else:
            # mode == "d1_macro" (Protocolo cerrado N=6 v3.1)
            df_enriched = MacroRatesProvider.align_causal_macro(df, macro_df) if not macro_df.empty else df
            primaries = _get_preregistered_primaries_for_asset(activo)
            logger.info(f"🚀 [v3.1 d1_macro] Torneo Cerrado N=6 para {activo} ({len(primaries)} primarias x 2 modelos)...")
            res = run_preregistered_v31_tournament(
                symbol=activo,
                df=df_enriched,
                primaries=primaries,
                results_dir=results_dir,
                candidate_names=("LOGISTIC_L2", "RANDOM_FOREST"),
                cusum_kappa=cusum_kappa,
                k_up=k_up,
                k_down=1.5,
                max_hold=10,
                enable_mda_pruning=enable_mda,
                export_onnx=True,
            )
            meta = res["model_bundle"]["meta_model"]
            summary_report[activo] = meta
            logger.info(
                f"✅ [{activo} d1_macro] Ganador: {res['winner_trial']} | CV LL={meta['cv_log_loss']:.4f} | "
                f"OOS LL={meta['oos_log_loss']:.4f} | ECE={meta['ece_oos']:.4f} | "
                f"OOS SR={meta['oos_net_sharpe']:.2f} vs Bench={meta['benchmark_sharpe']:.2f} | "
                f"DSR={meta['dsr_oos']:.3f} | CSCV PBO={meta['pbo_cscv']:.2%} | "
                f"Aprobado={meta['gate_approved']} | ONNX diff={meta['onnx_parity_max_diff']:.2e}"
            )

    out_summary_path = os.path.join(results_dir, f"summary_{mode}.json")
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)
    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Meta-Labeling Institucional v3.0 / v3.1")
    parser.add_argument(
        "--mode",
        type=str,
        default="d1_macro",
        choices=["d1_ohlcv", "d1_macro", "intraday_tib"],
        help="Modo de señal: línea base d1_ohlcv, señales macro d1_macro (N=6) o intradía TIB+VPIN",
    )
    parser.add_argument("--compare", action="store_true", help="Ejecutar benchmark comparativo d1_ohlcv vs d1_macro")
    parser.add_argument("--asset", type=str, default=None, help="Activo único a entrenar (ej. EURUSD)")
    parser.add_argument("--cusum-kappa", type=float, default=1.0, help="Multiplicador sigma para filtro CUSUM")
    parser.add_argument("--no-mda", action="store_true", help="Desactivar poda de features por MDA OOS")
    parser.add_argument("--refresh-macro", action="store_true", help="Forzar descarga fresca de FRED/yfinance")
    args = parser.parse_args()

    target_assets = [args.asset] if args.asset else ["EURUSD", "Oro", "SP500"]

    if args.compare:
        logger.info("⚖️ Ejecutando Benchmark Comparativo d1_ohlcv vs. d1_macro...")
        res_base = run_v3_training(target_assets, mode="d1_ohlcv", cusum_kappa=args.cusum_kappa, enable_mda=not args.no_mda)
        res_macro = run_v3_training(
            target_assets,
            mode="d1_macro",
            cusum_kappa=args.cusum_kappa,
            enable_mda=not args.no_mda,
            force_macro_refresh=args.refresh_macro,
        )
        print("\n=== COMPARATIVA OOS: d1_ohlcv (v3.0) vs. d1_macro (v3.1 N=6) ===")
        for act in target_assets:
            b = res_base.get(act, {})
            m = res_macro.get(act, {})
            print(
                f"{act:<8} | v3.0 OOS_LL={b.get('oos_log_loss', 'NA')} PBO={b.get('pbo_cscv', 'NA')} "
                f"|| v3.1 OOS_LL={m.get('oos_log_loss', 'NA')} PBO={m.get('pbo_cscv', 'NA')} "
                f"DSR={m.get('dsr_oos', 'NA')} Gate={m.get('gate_approved', False)}"
            )
    else:
        run_v3_training(
            target_assets,
            mode=args.mode,
            cusum_kappa=args.cusum_kappa,
            enable_mda=not args.no_mda,
            force_macro_refresh=args.refresh_macro,
        )
