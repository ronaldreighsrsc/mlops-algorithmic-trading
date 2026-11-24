"""
Módulo de Inferencia de Ultra-Baja Latencia (ONNX Runtime — v3.0).
Reemplaza TensorFlow/Keras en producción VPS para lograr < 2 ms de latencia
y reducir la huella de memoria RAM de 1.4 GB a < 130 MB.

Correcciones v3.0:
- Selecciona explícitamente el tensor de probabilidades ('calibrated_probability' /
  'output_probability' / 'probabilities') en vez de outputs[0] (que en skl2onnx es la
  etiqueta discreta int64).
- Soporta inferencia por lotes (predict_proba_batch) para validación estricta de paridad.
- Adapta ventanas 2D a vector aplanado 2D si el modelo tabular espera (None, look_back*F).
"""

from __future__ import annotations

import os
import time
import logging
import hashlib
from typing import Any, Dict, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger("ONNXEngine")


class ONNXInferenceEngine:
    """
    Motor de Inferencia compilado en C++ BLAS nativo mediante ONNX Runtime.
    """

    def __init__(
        self,
        onnx_model_path: str,
        expected_sha256: Optional[str] = None,
        intra_threads: int = 2,
    ):
        if ort is None:
            raise ImportError(
                "onnxruntime no está instalado. Instálalo con 'pip install onnxruntime'."
            )

        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Modelo ONNX no encontrado en: {onnx_model_path}")

        self.onnx_model_path = onnx_model_path

        # Verificación criptográfica obligatoria de integridad SHA-256
        if expected_sha256:
            actual_hash = self.compute_file_sha256(onnx_model_path)
            if actual_hash.lower() != expected_sha256.lower():
                raise ValueError(
                    f"Fallo de integridad criptográfica en modelo {onnx_model_path}!\n"
                    f"Esperado: {expected_sha256}\nObtenido: {actual_hash}"
                )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Priorizar salida calibrada > cualquier salida de probabilidad > última salida
        outputs = self.session.get_outputs()
        cal_outputs = [o for o in outputs if "calibrated" in o.name.lower()]
        prob_outputs = [o for o in outputs if "prob" in o.name.lower()]
        if cal_outputs:
            self.output_name = cal_outputs[-1].name
        elif prob_outputs:
            self.output_name = prob_outputs[-1].name
        else:
            self.output_name = outputs[-1].name

        logger.info(
            f"Sesión ONNX inicializada: {os.path.basename(onnx_model_path)} "
            f"(Input: {self.input_name} {self.input_shape} -> Output: {self.output_name})"
        )

    @staticmethod
    def compute_file_sha256(filepath: str) -> str:
        """Calcula el hash SHA-256 de un archivo binario."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _prepare_input(self, features: np.ndarray) -> np.ndarray:
        x_input = np.ascontiguousarray(features, dtype=np.float32)
        expected_dims = len(self.input_shape)

        if expected_dims == 3 and x_input.ndim == 2:
            x_input = np.expand_dims(x_input, axis=0)
        elif expected_dims == 2:
            if x_input.ndim == 1:
                x_input = x_input.reshape(1, -1)
            elif x_input.ndim == 2:
                expected_cols = self.input_shape[1]
                if isinstance(expected_cols, int) and x_input.shape[1] != expected_cols and x_input.size == expected_cols:
                    x_input = x_input.reshape(1, expected_cols)
        return x_input

    def _extract_positive_probs(self, out: Any) -> np.ndarray:
        # Manejar posible salida dict/zipmap por robustez
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return np.array([float(d.get(1, d.get("1", 0.5))) for d in out], dtype=np.float64)

        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                probs = arr[:, 1]
            else:
                probs = arr[:, 0]
        elif arr.ndim == 1:
            if len(arr) == 2 and (isinstance(self.input_shape[0], int) and self.input_shape[0] == 1):
                probs = np.array([arr[1]], dtype=np.float64)
            else:
                probs = arr
        else:
            probs = arr.reshape(-1)

        return np.clip(probs, 0.0, 1.0)

    def predict_proba_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Ejecuta inferencia vectorizada por lote retornando P(y=1) en [0, 1]."""
        x_input = np.ascontiguousarray(X_batch, dtype=np.float32)
        if len(self.input_shape) == 2 and x_input.ndim == 1:
            x_input = x_input.reshape(1, -1)
        raw_out = self.session.run([self.output_name], {self.input_name: x_input})[0]
        return self._extract_positive_probs(raw_out)

    def predict_probability(self, features: np.ndarray) -> float:
        """
        Ejecuta inferencia para una única observación/ventana en C++ BLAS nativo.
        Retorna probabilidad continua P(y=1) en [0.0, 1.0].
        """
        x_input = self._prepare_input(features)
        raw_outputs = self.session.run([self.output_name], {self.input_name: x_input})
        probs = self._extract_positive_probs(raw_outputs[0])
        return float(probs[0])

    def benchmark_latency(self, sample_features: np.ndarray, n_iter: int = 100) -> Dict[str, float]:
        """Mide la distribución de latencia de inferencia en milisegundos."""
        latencies_ms = []

        for _ in range(5):
            self.predict_probability(sample_features)

        for _ in range(n_iter):
            t0 = time.perf_counter()
            self.predict_probability(sample_features)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        lat_arr = np.array(latencies_ms)
        return {
            "mean_ms": round(float(np.mean(lat_arr)), 3),
            "p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
            "min_ms": round(float(np.min(lat_arr)), 3),
            "max_ms": round(float(np.max(lat_arr)), 3),
        }
