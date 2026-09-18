"""
Tests unitarios para el Motor de Inferencia ONNX y Gobernanza Criptográfica.
"""

import os
import tempfile
import pytest
import numpy as np
from src.models.onnx_exporter import (
    create_synthetic_onnx_model,
    compute_sha256,
    update_onnx_manifest
)
from src.execution.onnx_inference_engine import ONNXInferenceEngine


@pytest.fixture
def temp_onnx_models_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tabular_path = os.path.join(tmp_dir, "test_tabular.onnx")
        seq_path = os.path.join(tmp_dir, "test_sequence.onnx")

        create_synthetic_onnx_model(tabular_path, input_dim=5, is_sequence=False)
        create_synthetic_onnx_model(seq_path, input_dim=4, is_sequence=True, seq_len=10)

        yield tmp_dir, tabular_path, seq_path


def test_tabular_onnx_inference(temp_onnx_models_dir):
    """Verifica inferencia sobre modelo tabular (ej. XGBoost / Random Forest)."""
    _, tabular_path, _ = temp_onnx_models_dir
    sha256 = compute_sha256(tabular_path)

    engine = ONNXInferenceEngine(tabular_path, expected_sha256=sha256)

    # Inferencia con array 1D
    sample_1d = np.array([0.5, -0.2, 1.1, 0.0, -0.8], dtype=np.float32)
    prob_1d = engine.predict_probability(sample_1d)
    assert isinstance(prob_1d, float)
    assert 0.0 <= prob_1d <= 1.0

    # Inferencia con array 2D (1, 5)
    sample_2d = sample_1d.reshape(1, -1)
    prob_2d = engine.predict_probability(sample_2d)
    assert abs(prob_1d - prob_2d) < 1e-6


def test_sequence_onnx_inference(temp_onnx_models_dir):
    """Verifica inferencia sobre modelo de secuencias (ej. BiLSTM / LSTM)."""
    _, _, seq_path = temp_onnx_models_dir
    engine = ONNXInferenceEngine(seq_path)

    # Inferencia con tensor 2D (seq_len=10, features=4)
    sample_seq = np.random.normal(0, 1, size=(10, 4)).astype(np.float32)
    prob = engine.predict_probability(sample_seq)

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_sha256_checksum_mismatch_raises_error(temp_onnx_models_dir):
    """Un hash alterado o incorrecto debe abortar la carga del modelo con ValueError."""
    _, tabular_path, _ = temp_onnx_models_dir
    fake_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    with pytest.raises(ValueError, match="Fallo de integridad criptográfica"):
        ONNXInferenceEngine(tabular_path, expected_sha256=fake_sha)


def test_onnx_latency_benchmark(temp_onnx_models_dir):
    """Verifica que el benchmark de latencia reporte métricas en milisegundos."""
    _, tabular_path, _ = temp_onnx_models_dir
    engine = ONNXInferenceEngine(tabular_path)

    sample = np.ones((1, 5), dtype=np.float32)
    metrics = engine.benchmark_latency(sample, n_iter=50)

    assert "mean_ms" in metrics
    assert "p50_ms" in metrics
    assert "p99_ms" in metrics
    assert metrics["mean_ms"] >= 0.0
    # Inferencia C++ nativa en CPU para un grafo ligero típicamente toma < 2.0 ms
    assert metrics["mean_ms"] < 10.0


def test_onnx_manifest_generation(temp_onnx_models_dir):
    """Verifica la generación del archivo onnx_manifest.json."""
    tmp_dir, _, _ = temp_onnx_models_dir
    manifest_path = os.path.join(tmp_dir, "onnx_manifest.json")

    manifest = update_onnx_manifest(tmp_dir, manifest_path)
    assert "test_tabular.onnx" in manifest
    assert "test_sequence.onnx" in manifest
    assert os.path.exists(manifest_path)
