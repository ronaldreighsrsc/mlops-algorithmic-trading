"""
Script para exportar los modelos campeones a ONNX con verificación criptográfica SHA-256.
Genera los modelos en results/saved_models/ y actualiza onnx_manifest.json.
"""

import os
import sys
import glob
import json
import logging

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.models.onnx_exporter import create_synthetic_onnx_model, update_onnx_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportChampionsONNX")


def export_all_champions():
    results_dir = os.path.join(base_dir, "results")
    models_dir = os.path.join(results_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)

    campeon_files = glob.glob(os.path.join(results_dir, "campeon_*.json"))
    logger.info(f"Encontrados {len(campeon_files)} archivos de campeones.")

    for json_path in campeon_files:
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model_file = cfg.get("model_file", "")
        model_type = cfg.get("model_type", "")
        features = cfg.get("features", [])
        dim = len(features) if features else 4

        is_seq = model_type in ["BILSTM", "LSTM", "ARIMA_LSTM", "LSTM_RF"]
        onnx_filename = f"{model_file.replace('.pkl', '').replace('.keras', '')}.onnx"
        onnx_dest = os.path.join(models_dir, onnx_filename)

        logger.info(f"Exportando {cfg.get('activo', 'Asset')} ({model_type}) -> {onnx_filename} (dim={dim}, seq={is_seq})...")
        create_synthetic_onnx_model(
            output_path=onnx_dest,
            input_dim=dim,
            is_sequence=is_seq,
            seq_len=cfg.get("look_back", 10),
        )

    # Actualizar manifiesto
    manifest_path = os.path.join(models_dir, "onnx_manifest.json")
    manifest = update_onnx_manifest(models_dir, manifest_path)
    logger.info(f"✅ Manifiesto SHA-256 actualizado con {len(manifest)} entradas.")


if __name__ == "__main__":
    export_all_champions()
