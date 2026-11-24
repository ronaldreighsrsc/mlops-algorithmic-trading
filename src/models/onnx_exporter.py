"""
Módulo de Exportación Real y Gobernanza Criptográfica de Modelos ONNX (v3.0).

Características v3.0:
1. Convierte Pipelines reales completos (SimpleImputer + StandardScaler + Classifier)
   de scikit-learn, XGBoost y LightGBM a grafos ONNX autocontenidos.
2. Embebe nodos exactos de calibración Platt (log-odds) directamente en el grafo ONNX
   cuando se suministra un PurgedPlattCalibrator.
3. Aplica una compuerta obligatoria de paridad numérica (< 1e-5) contra el modelo nativo
   antes de aprobar el archivo .onnx y actualizar onnx_manifest.json (SHA-256).
4. create_synthetic_onnx_model() queda restringido únicamente a pruebas unitarias.
"""

from __future__ import annotations

import os
import json
import logging
import hashlib
import warnings
from typing import Dict, Optional, Tuple, Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.execution.onnx_inference_engine import ONNXInferenceEngine
from src.models.calibration import PurgedPlattCalibrator

logger = logging.getLogger("ONNXExporter")

_CONVERTERS_REGISTERED = False


def compute_sha256(filepath: str) -> str:
    """Calcula el checksum SHA-256 de un archivo binario."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _register_tree_converters() -> None:
    """Registra los convertidores oficiales de XGBoost y LightGBM en skl2onnx."""
    global _CONVERTERS_REGISTERED
    if _CONVERTERS_REGISTERED:
        return

    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

    try:
        import xgboost as xgb
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

        update_registered_converter(
            xgb.XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
    except Exception as exc:
        logger.warning(f"No se pudo registrar convertidor XGBoost ONNX: {exc}")

    try:
        import lightgbm as lgb
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

        update_registered_converter(
            lgb.LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
    except Exception as exc:
        logger.warning(f"No se pudo registrar convertidor LightGBM ONNX: {exc}")

    _CONVERTERS_REGISTERED = True


def _append_platt_calibration_nodes(
    model_proto: Any,
    calibrator: PurgedPlattCalibrator,
) -> Any:
    """
    Agrega calibración Platt exacta en log-odds al final de un grafo ONNX clasificador:
        p1 = probs[:, 1:2]
        p_clip = Clip(p1, eps, 1 - eps)
        logit = Log(p_clip / (1 - p_clip))
        p_cal = Sigmoid(coef * logit + intercept)
    """
    import onnx
    from onnx import helper, TensorProto

    graph = model_proto.graph
    prob_outputs = [o for o in graph.output if "prob" in o.name.lower()]
    if not prob_outputs:
        raise ValueError("El grafo ONNX base no expone una salida de probabilidades.")

    base_prob_name = prob_outputs[-1].name

    # Inicializadores constantes
    eps = float(calibrator.eps)
    initializers = [
        helper.make_tensor("platt_slice_starts", TensorProto.INT64, [1], [1]),
        helper.make_tensor("platt_slice_ends", TensorProto.INT64, [1], [2]),
        helper.make_tensor("platt_slice_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("platt_eps", TensorProto.FLOAT, [], [eps]),
        helper.make_tensor("platt_one_minus_eps", TensorProto.FLOAT, [], [1.0 - eps]),
        helper.make_tensor("platt_one", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("platt_coef", TensorProto.FLOAT, [], [float(calibrator.coef_)]),
        helper.make_tensor("platt_intercept", TensorProto.FLOAT, [], [float(calibrator.intercept_)]),
        helper.make_tensor("platt_z_min", TensorProto.FLOAT, [], [-30.0]),
        helper.make_tensor("platt_z_max", TensorProto.FLOAT, [], [30.0]),
    ]
    graph.initializer.extend(initializers)

    nodes = [
        helper.make_node(
            "Slice",
            inputs=[base_prob_name, "platt_slice_starts", "platt_slice_ends", "platt_slice_axes"],
            outputs=["platt_p1"],
        ),
        helper.make_node(
            "Clip",
            inputs=["platt_p1", "platt_eps", "platt_one_minus_eps"],
            outputs=["platt_p_clip"],
        ),
        helper.make_node(
            "Sub",
            inputs=["platt_one", "platt_p_clip"],
            outputs=["platt_denom"],
        ),
        helper.make_node(
            "Div",
            inputs=["platt_p_clip", "platt_denom"],
            outputs=["platt_odds"],
        ),
        helper.make_node(
            "Log",
            inputs=["platt_odds"],
            outputs=["platt_logit"],
        ),
        helper.make_node(
            "Mul",
            inputs=["platt_logit", "platt_coef"],
            outputs=["platt_scaled"],
        ),
        helper.make_node(
            "Add",
            inputs=["platt_scaled", "platt_intercept"],
            outputs=["platt_z_raw"],
        ),
        helper.make_node(
            "Clip",
            inputs=["platt_z_raw", "platt_z_min", "platt_z_max"],
            outputs=["platt_z"],
        ),
        helper.make_node(
            "Sigmoid",
            inputs=["platt_z"],
            outputs=["calibrated_probability"],
        ),
    ]
    graph.node.extend(nodes)

    cal_output = helper.make_tensor_value_info(
        "calibrated_probability", TensorProto.FLOAT, [None, 1]
    )
    graph.output.append(cal_output)
    onnx.checker.check_model(model_proto)
    return model_proto


def export_sklearn_pipeline_to_onnx(
    pipeline: Any,
    output_path: str,
    X_parity_sample: np.ndarray,
    calibrator: Optional[PurgedPlattCalibrator] = None,
    max_parity_tol: float = 1e-5,
) -> Tuple[str, str, float]:
    """
    Convierte un Pipeline real (Imputer + StandardScaler + Classifier) a ONNX y verifica
    paridad numérica estricta (< max_parity_tol) contra el modelo nativo.

    Retorna: (output_path, sha256, max_abs_diff)
    """
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    _register_tree_converters()

    X_eval = np.asarray(X_parity_sample, dtype=np.float32)
    if X_eval.ndim == 1:
        X_eval = X_eval.reshape(1, -1)
    if X_eval.shape[0] == 0:
        raise ValueError("X_parity_sample no puede estar vacío para verificar paridad ONNX.")

    n_features = int(X_eval.shape[1])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=[("float_input", FloatTensorType([None, n_features]))],
            target_opset={"": 17, "ai.onnx.ml": 3},
            options={id(pipeline): {"zipmap": False}},
        )

    if calibrator is not None and calibrator.is_fitted_:
        onnx_model = _append_platt_calibration_nodes(onnx_model, calibrator)

    onnx.save(onnx_model, output_path)

    # Compuerta obligatoria de paridad numérica Train/Serve
    p_native_raw = pipeline.predict_proba(X_eval)[:, 1]
    if calibrator is not None and calibrator.is_fitted_:
        p_native = calibrator.transform(p_native_raw)
    else:
        p_native = p_native_raw

    engine = ONNXInferenceEngine(output_path)
    p_onnx = engine.predict_proba_batch(X_eval)
    max_diff = float(np.max(np.abs(np.asarray(p_native, dtype=np.float64) - np.asarray(p_onnx, dtype=np.float64))))

    if not np.isfinite(max_diff) or max_diff >= max_parity_tol:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(
            f"Fallo en compuerta de paridad ONNX para {os.path.basename(output_path)}: "
            f"max_abs_diff={max_diff:.2e} >= tolerancia {max_parity_tol:.2e}. "
            f"Artefacto eliminado para evitar despliegue corrupto."
        )

    sha256 = compute_sha256(output_path)
    logger.info(
        f"✅ ONNX real exportado y validado: {os.path.basename(output_path)} "
        f"(features={n_features}, max_diff={max_diff:.2e}, sha256={sha256[:12]}...)"
    )
    return output_path, sha256, max_diff


def export_legacy_trainer_pkl_to_onnx(
    pkl_path: str,
    output_path: str,
    X_parity_sample: Optional[np.ndarray] = None,
    max_parity_tol: float = 1e-5,
) -> Tuple[str, str, float]:
    """
    Convierte un artefacto legacy guardado por XGBoostTrainer o RandomForestTrainer
    ({'scaler': StandardScaler, 'master_model': model, 'look_back': int}) a un grafo
    ONNX real con StandardScaler embebido y paridad verificada.
    """
    state = joblib.load(pkl_path)
    if not isinstance(state, dict) or "master_model" not in state or "scaler" not in state:
        raise ValueError(
            f"El archivo {pkl_path} no contiene un trainer tabular ('scaler' + 'master_model')."
        )

    scaler = state["scaler"]
    model = state["master_model"]
    n_features = int(getattr(scaler, "n_features_in_", 0))
    if n_features <= 0:
        raise ValueError(f"El StandardScaler de {pkl_path} no está ajustado (n_features_in_ inválido).")

    pipe = Pipeline([("scaler", scaler), ("clf", model)])

    if X_parity_sample is None:
        # Generar muestras realistas alrededor de la distribución entrenada del scaler
        rng = np.random.default_rng(42)
        mean = getattr(scaler, "mean_", np.zeros(n_features))
        scale = getattr(scaler, "scale_", np.ones(n_features))
        X_parity_sample = (rng.normal(size=(100, n_features)) * scale + mean).astype(np.float32)

    return export_sklearn_pipeline_to_onnx(
        pipeline=pipe,
        output_path=output_path,
        X_parity_sample=X_parity_sample,
        calibrator=None,
        max_parity_tol=max_parity_tol,
    )


def create_synthetic_onnx_model(
    output_path: str,
    input_dim: int = 3,
    is_sequence: bool = False,
    seq_len: int = 10,
) -> str:
    """
    [SOLO PRUEBAS UNITARIAS / BENCHMARKS]
    Construye un grafo ONNX mínimo para tests unitarios de infraestructura y latencia.
    NUNCA usar en pipelines de producción; usar export_sklearn_pipeline_to_onnx().
    """
    import onnx
    from onnx import helper, TensorProto

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if is_sequence:
        input_shape = [1, seq_len, input_dim]
        total_dim = seq_len * input_dim
    else:
        input_shape = [1, input_dim]
        total_dim = input_dim

    X = helper.make_tensor_value_info("float_input", TensorProto.FLOAT, input_shape)
    Y = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 1])

    nodes = []
    if is_sequence:
        nodes.append(
            helper.make_node("Flatten", inputs=["float_input"], outputs=["flattened"], axis=1)
        )
        dense_input = "flattened"
    else:
        dense_input = "float_input"

    w_vals = np.full((total_dim, 1), 0.1 / total_dim, dtype=np.float32)
    b_vals = np.array([0.0], dtype=np.float32)

    W = helper.make_tensor("W", TensorProto.FLOAT, [total_dim, 1], w_vals.flatten().tolist())
    B = helper.make_tensor("B", TensorProto.FLOAT, [1], b_vals.tolist())

    nodes.extend([
        helper.make_node("MatMul", inputs=[dense_input, "W"], outputs=["logits_raw"]),
        helper.make_node("Add", inputs=["logits_raw", "B"], outputs=["logits"]),
        helper.make_node("Sigmoid", inputs=["logits"], outputs=["probabilities"]),
    ])

    graph_def = helper.make_graph(
        nodes=nodes,
        name="QuantSyntheticTestGraph",
        inputs=[X],
        outputs=[Y],
        initializer=[W, B],
    )

    model_def = helper.make_model(graph_def, producer_name="QuantBotTestHelper")
    model_def.opset_import[0].version = 17

    onnx.save(model_def, output_path)
    return output_path


def update_onnx_manifest(models_dir: str, manifest_path: Optional[str] = None) -> Dict[str, str]:
    """Escanea todos los archivos .onnx en models_dir y genera el manifiesto SHA-256."""
    if manifest_path is None:
        manifest_path = os.path.join(models_dir, "onnx_manifest.json")

    manifest = {}
    for fname in sorted(os.listdir(models_dir)):
        if fname.endswith(".onnx"):
            fpath = os.path.join(models_dir, fname)
            manifest[fname] = compute_sha256(fpath)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    logger.info(f"Manifiesto ONNX actualizado con {len(manifest)} modelos en {manifest_path}")
    return manifest
