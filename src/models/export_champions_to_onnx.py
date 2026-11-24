"""
Script de exportación REAL de modelos campeones a ONNX con verificación de paridad (< 1e-5)
y gobernanza criptográfica SHA-256 (v3.0).

IMPORTANTE: Ya NO utiliza grafos sintéticos dummy. Todo artefacto .onnx exportado refleja
fielmente los pesos reales del Pipeline entrenado (Scaler + Clasificador + Calibrador).
"""

from __future__ import annotations

import os
import sys
import glob
import json
import logging
from typing import Dict

import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.models.onnx_exporter import (
    export_sklearn_pipeline_to_onnx,
    export_legacy_trainer_pkl_to_onnx,
    update_onnx_manifest,
)
from src.models.calibration import PurgedPlattCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportChampionsONNX")


def export_all_champions(results_dir: str | None = None) -> Dict[str, dict]:
    """
    Exporta a ONNX real todos los campeones tabulares válidos encontrados en results/campeon_*.json.
    Elimina cualquier .onnx sintético obsoleto si el campeón no soporta conversión real verificada.
    """
    res_dir = results_dir or os.path.join(base_dir, "results")
    models_dir = os.path.join(res_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    campeon_files = sorted(glob.glob(os.path.join(res_dir, "campeon_*.json")))
    logger.info(f"Encontrados {len(campeon_files)} archivos de campeones en {res_dir}.")

    report: Dict[str, dict] = {}

    for json_path in campeon_files:
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        activo = cfg.get("activo", os.path.basename(json_path))
        schema_ver = str(cfg.get("schema_version", "2.0"))

        # --- Caso v3.0: ModelBundle con Pipeline + PurgedPlattCalibrator ---
        if schema_ver.startswith("3.") and "meta_model" in cfg:
            meta_cfg = cfg["meta_model"]
            bundle_pkl = os.path.join(models_dir, meta_cfg.get("bundle_pkl", ""))
            onnx_name = meta_cfg.get("onnx_file", f"meta_{activo.lower()}.onnx")
            onnx_dest = os.path.join(models_dir, onnx_name)

            if not os.path.exists(bundle_pkl):
                logger.warning(f"[{activo}] Bundle v3.0 no encontrado: {bundle_pkl}. Omitido.")
                continue

            bundle = joblib.load(bundle_pkl)
            pipe = bundle["pipeline"]
            cal_dict = bundle.get("calibrator")
            calibrator = PurgedPlattCalibrator.from_dict(cal_dict) if cal_dict else None
            X_parity = bundle["X_parity_sample"]

            _, sha256, max_diff = export_sklearn_pipeline_to_onnx(
                pipeline=pipe,
                output_path=onnx_dest,
                X_parity_sample=X_parity,
                calibrator=calibrator,
            )

            cfg["meta_model"]["sha256"] = sha256
            cfg["meta_model"]["onnx_parity_max_diff"] = max_diff
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)

            report[activo] = {"status": "EXPORTED_V3", "onnx": onnx_name, "max_diff": max_diff, "sha256": sha256}
            continue

        # --- Caso Legacy v1/v2: XGBOOST / RANDOM_FOREST (.pkl con scaler + master_model) ---
        model_file = cfg.get("model_file", "")
        model_type = str(cfg.get("model_type", "")).upper()
        pkl_path = os.path.join(models_dir, model_file)
        onnx_filename = f"{model_file.replace('.pkl', '').replace('.keras', '')}.onnx"
        onnx_dest = os.path.join(models_dir, onnx_filename)

        if model_type in ("XGBOOST", "RANDOM_FOREST", "LIGHTGBM", "LOGISTIC_REGRESSION") and os.path.exists(pkl_path):
            try:
                _, sha256, max_diff = export_legacy_trainer_pkl_to_onnx(pkl_path, onnx_dest)
                report[activo] = {
                    "status": "EXPORTED_LEGACY_REAL",
                    "onnx": onnx_filename,
                    "max_diff": max_diff,
                    "sha256": sha256,
                }
            except Exception as exc:
                logger.error(f"[{activo}] Error exportando {model_file} a ONNX real: {exc}")
                report[activo] = {"status": "ERROR", "error": str(exc)}
        else:
            # Prohibido emitir modelos sintéticos falsos en producción
            if os.path.exists(onnx_dest) and os.path.getsize(onnx_dest) < 2048:
                os.remove(onnx_dest)
                logger.warning(
                    f"[{activo}] Eliminado artefacto ONNX sintético legacy ({onnx_filename}). "
                    f"Reentrenar con pipeline tabular v3.0 (`src/main_training_v3.py`) para exportar ONNX real."
                )
            else:
                logger.warning(
                    f"[{activo}] Modelo legacy '{model_type}' no soportado por exportador tabular real. "
                    f"Ejecute el entrenamiento v3.0 (Meta-Labeling Tabular)."
                )
            report[activo] = {"status": "SKIPPED_NON_TABULAR_LEGACY", "model_type": model_type}

    manifest_path = os.path.join(models_dir, "onnx_manifest.json")
    manifest = update_onnx_manifest(models_dir, manifest_path)
    logger.info(f"✅ Manifiesto SHA-256 actualizado con {len(manifest)} modelos reales verificados.")
    return report


if __name__ == "__main__":
    export_all_champions()
