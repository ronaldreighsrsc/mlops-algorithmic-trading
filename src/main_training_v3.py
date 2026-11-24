"""
Pipeline Maestro de Entrenamiento Meta-Labeling Institucional v3.0.

Ejecución:
    python src/main_training_v3.py
    python src/main_training_v3.py --asset EURUSD --cusum-kappa 1.0
"""

from __future__ import annotations

import os
import sys
import argparse
import logging

import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.models.meta_trainer_v3 import MetaModelTrainerV3
from src.strategies.primary_strategies import get_primary_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MainTrainingV3")


def run_v3_training(assets: list[str], cusum_kappa: float = 1.0, enable_mda: bool = True) -> None:
    results_dir = os.path.join(base_dir, "results")
    processed_dir = os.path.join(base_dir, "data", "processed")

    for activo in assets:
        proc_path = os.path.join(processed_dir, f"{activo}_processed.csv")
        if not os.path.exists(proc_path):
            logger.warning(f"[{activo}] Dataset procesado no encontrado en {proc_path}. Omitido.")
            continue

        df = pd.read_csv(proc_path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)

        bilateral = activo.upper().startswith("EUR") or activo.upper().startswith("GBP") or activo.upper().startswith("USD")
        k_up = 2.5 if activo.lower().startswith("oro") or activo.upper().startswith("XAU") else 2.0
        primary = get_primary_strategy("TrendPullbackPrimary", sma_slow=200, ema_fast=20, ema_slow=50, bilateral=bilateral)

        logger.info(f"🚀 [v3.0] Entrenando Meta-Modelo Tabular Calibrado para {activo} (bilateral={bilateral})...")
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
        logger.info(
            f"✅ [{activo}] Ganador: {res['winner']} | CV LogLoss={meta['cv_log_loss']:.4f} | "
            f"OOS Brier={meta['brier_score_oos']:.4f} | ECE={meta['ece_oos']:.4f} | "
            f"DSR={meta['dsr_oos']:.3f} | CSCV PBO={meta['pbo_cscv']:.2%} | "
            f"ONNX max_diff={meta['onnx_parity_max_diff']:.2e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Meta-Labeling Institucional v3.0")
    parser.add_argument("--asset", type=str, default=None, help="Activo único a entrenar (ej. EURUSD)")
    parser.add_argument("--cusum-kappa", type=float, default=1.0, help="Multiplicador sigma para filtro CUSUM")
    parser.add_argument("--no-mda", action="store_true", help="Desactivar poda de features por MDA OOS")
    args = parser.parse_args()

    target_assets = [args.asset] if args.asset else ["EURUSD", "Oro", "SP500"]
    run_v3_training(target_assets, cusum_kappa=args.cusum_kappa, enable_mda=not args.no_mda)
