"""
Módulo de Inferencia de Ultra-Baja Latencia (ONNX Runtime).
Reemplaza TensorFlow/Keras en producción VPS para lograr < 2 ms de latencia
y reducir la huella de memoria RAM de 1.4 GB a < 130 MB.
"""

import os
import time
import logging
import hashlib
import numpy as np
from typing import Dict, Optional, Tuple, Any

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

        # Verificación criptográfica de integridad SHA256 si se suministra
        if expected_sha256:
            actual_hash = self.compute_file_sha256(onnx_model_path)
            if actual_hash.lower() != expected_sha256.lower():
                raise ValueError(
                    f"Fallo de integridad criptográfica en modelo {onnx_model_path}!\n"
                    f"Esperado: {expected_sha256}\nObtenido: {actual_hash}"
                )

        # Configurar SessionOptions optimizadas para VPS CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Cargar sesión en CPUExecutionProvider
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            f"Sesión ONNX inicializada: {os.path.basename(onnx_model_path)} "
            f"(Input: {self.input_name} {self.input_shape})"
        )

    @staticmethod
    def compute_file_sha256(filepath: str) -> str:
        """Calcula el hash SHA-256 de un archivo binario."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def predict_probability(self, features: np.ndarray) -> float:
        """
        Ejecuta la inferencia en C++ BLAS nativo.
        Acepta features como ndarray (2D o 3D según el modelo).
        Retorna: Probabilidad continua de clase positiva P(TP) en [0.0, 1.0].
        """
        # Asegurar tipo de datos float32 y contigüidad estricta en memoria
        x_input = np.ascontiguousarray(features, dtype=np.float32)

        # Adaptar dimensiones si el modelo espera shape específico
        expected_dims = len(self.input_shape)
        if x_input.ndim < expected_dims:
            if expected_dims == 3 and x_input.ndim == 2:
                # Expandir batch o secuencia (1, seq_len, features)
                x_input = np.expand_dims(x_input, axis=0)
            elif expected_dims == 2 and x_input.ndim == 1:
                x_input = x_input.reshape(1, -1)

        raw_outputs = self.session.run([self.output_name], {self.input_name: x_input})

        # Extraer la probabilidad de forma agnóstica a la arquitectura del grafo
        out = raw_outputs[0]

        # Caso: Array 3D o 2D con salida binaria (batch, 2) o sigmoide (batch, 1)
        if out.ndim >= 2:
            if out.shape[-1] == 2:
                prob = float(out[0][1])  # Clase 1 en clasificador multiclase
            else:
                prob = float(out[0][0])
        elif out.ndim == 1:
            prob = float(out[1]) if len(out) == 2 else float(out[0])
        else:
            prob = float(out)

        # Truncar a rango válido de probabilidad [0.0, 1.0]
        return max(0.0, min(1.0, prob))

    def benchmark_latency(self, sample_features: np.ndarray, n_iter: int = 100) -> Dict[str, float]:
        """
        Mide la distribución de latencia de inferencia en milisegundos.
        """
        latencies_ms = []

        # Warmup (5 pasadas)
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
