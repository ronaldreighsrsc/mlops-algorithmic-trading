"""
Módulo de Exportación y Gobernanza de Modelos ONNX (v2.0).
Convierte modelos entrenados a formato ONNX (.onnx) y genera un manifiesto
criptográfico con checksums SHA-256 para validación estricta de integridad en VPS.
"""

import os
import json
import logging
import hashlib
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger("ONNXExporter")


def compute_sha256(filepath: str) -> str:
    """Calcula el checksum SHA-256 de un archivo."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_synthetic_onnx_model(
    output_path: str,
    input_dim: int = 3,
    is_sequence: bool = False,
    seq_len: int = 10,
) -> str:
    """
    Construye un grafo ONNX válido y ligero utilizando la API nativa de 'onnx.helper'.
    Aplica una transformación afín + sigmoide para retornar una probabilidad en [0, 1].
    Útil para pruebas unitarias, benchmarks y validación sin dependencias externas pesadas.
    """
    import onnx
    from onnx import helper, TensorProto

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if is_sequence:
        # Input shape: (batch_size, seq_len, input_dim)
        input_shape = [1, seq_len, input_dim]
        # Reducir secuencia sumando o aplanando
        total_dim = seq_len * input_dim
    else:
        # Input shape: (batch_size, input_dim)
        input_shape = [1, input_dim]
        total_dim = input_dim

    # Inputs y Outputs del grafo
    X = helper.make_tensor_value_info("float_input", TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 1])

    nodes = []

    if is_sequence:
        # Aplanar (1, seq_len, input_dim) a (1, total_dim)
        flatten_node = helper.make_node(
            "Flatten",
            inputs=["float_input"],
            outputs=["flattened"],
            axis=1,
        )
        nodes.append(flatten_node)
        dense_input = "flattened"
    else:
        dense_input = "float_input"

    # Pesos constantes W (total_dim, 1) y sesgo B (1,)
    w_vals = np.full((total_dim, 1), 0.1 / total_dim, dtype=np.float32)
    b_vals = np.array([0.0], dtype=np.float32)

    W = helper.make_tensor("W", TensorProto.FLOAT, [total_dim, 1], w_vals.flatten().tolist())
    B = helper.make_tensor("B", TensorProto.FLOAT, [1], b_vals.tolist())

    # Multiplicación matricial (MatMul) y suma (Add)
    matmul_node = helper.make_node(
        "MatMul",
        inputs=[dense_input, "W"],
        outputs=["logits_raw"],
    )
    add_node = helper.make_node(
        "Add",
        inputs=["logits_raw", "B"],
        outputs=["logits"],
    )
    sigmoid_node = helper.make_node(
        "Sigmoid",
        inputs=["logits"],
        outputs=["probabilities"],
    )
    nodes.extend([matmul_node, add_node, sigmoid_node])

    # Construir grafo y modelo ONNX
    graph_def = helper.make_graph(
        nodes=nodes,
        name="QuantModelGraph",
        inputs=[X],
        outputs=[Y],
        initializer=[W, B],
    )

    model_def = helper.make_model(graph_def, producer_name="QuantBotV2Exporter")
    model_def.opset_import[0].version = 17

    onnx.save(model_def, output_path)
    logger.info(f"Modelo sintético ONNX generado con éxito en: {output_path}")
    return output_path


def update_onnx_manifest(models_dir: str, manifest_path: Optional[str] = None) -> Dict[str, str]:
    """
    Escanea todos los archivos .onnx en models_dir y genera el manifiesto SHA-256.
    """
    if manifest_path is None:
        manifest_path = os.path.join(models_dir, "onnx_manifest.json")

    manifest = {}
    for fname in os.listdir(models_dir):
        if fname.endswith(".onnx"):
            fpath = os.path.join(models_dir, fname)
            manifest[fname] = compute_sha256(fpath)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    logger.info(f"Manifiesto ONNX actualizado con {len(manifest)} modelos en {manifest_path}")
    return manifest
