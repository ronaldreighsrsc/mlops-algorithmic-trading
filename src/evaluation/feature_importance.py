"""
Importancia de Características Out-of-Sample vía MDA Purgado (AFML Cap. 8).

A diferencia de MDI (feature_importances_ in-sample de sklearn/XGBoost), que sufre
sesgo de sustitución y sobreajusta a variables con alta cardinalidad, MDA evalúa el
incremento real de Log-Loss al permutar cada feature exclusivamente sobre folds de
validación purgados (PurgedKFold). Toda variable con delta_logloss <= 0 es ruido puro.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.validation.purged_cv import PurgedKFold


def compute_purged_mda(
    pipeline_factory: Callable[[], object],
    X_df: pd.DataFrame,
    y: np.ndarray,
    t0: pd.Series | np.ndarray,
    t1: pd.Series | np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula Mean Decrease Accuracy / Log-Loss (MDA) purgado out-of-sample.

    Importancia positiva (> 0): permutar la columna EMPEORÓ el Log-Loss OOS
    (la variable aporta señal real).
    Importancia <= 0: permutar la columna no afectó o MEJORÓ el Log-Loss OOS
    (ruido puro -> candidata a poda).
    """
    feature_names = list(X_df.columns)
    X_arr = X_df.values.astype(np.float64)
    y_arr = np.asarray(y, dtype=np.int32).ravel()
    w_arr = (
        np.ones(len(y_arr), dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64).ravel()
    )

    cv = PurgedKFold(n_splits=n_splits, pct_embargo=pct_embargo)
    rng = np.random.default_rng(random_state)

    fold_deltas = {col: [] for col in feature_names}

    for train_idx, val_idx in cv.split(t0, t1):
        if len(train_idx) < 20 or len(val_idx) < 5:
            continue
        if len(np.unique(y_arr[train_idx])) < 2 or len(np.unique(y_arr[val_idx])) < 2:
            continue

        pipe = pipeline_factory()
        fit_kwargs = {}
        # Pasar sample_weight al último paso del Pipeline si es un Pipeline
        if hasattr(pipe, "steps") and pipe.steps:
            last_step_name = pipe.steps[-1][0]
            fit_kwargs[f"{last_step_name}__sample_weight"] = w_arr[train_idx]
        else:
            fit_kwargs["sample_weight"] = w_arr[train_idx]

        try:
            pipe.fit(X_arr[train_idx], y_arr[train_idx], **fit_kwargs)
        except TypeError:
            pipe.fit(X_arr[train_idx], y_arr[train_idx])

        base_probs = np.clip(pipe.predict_proba(X_arr[val_idx])[:, 1], 1e-6, 1.0 - 1e-6)
        base_ll = float(log_loss(y_arr[val_idx], base_probs, sample_weight=w_arr[val_idx], labels=[0, 1]))

        X_val_orig = X_arr[val_idx]
        for col_idx, col_name in enumerate(feature_names):
            rep_deltas = []
            for _ in range(n_repeats):
                X_perm = X_val_orig.copy()
                X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])
                perm_probs = np.clip(pipe.predict_proba(X_perm)[:, 1], 1e-6, 1.0 - 1e-6)
                perm_ll = float(log_loss(y_arr[val_idx], perm_probs, sample_weight=w_arr[val_idx], labels=[0, 1]))
                rep_deltas.append(perm_ll - base_ll)
            fold_deltas[col_name].append(float(np.mean(rep_deltas)))

    rows = []
    for col_name in feature_names:
        vals = fold_deltas[col_name]
        mean_imp = float(np.mean(vals)) if vals else 0.0
        std_imp = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append({
            "feature": col_name,
            "mean_importance": mean_imp,
            "std_error": std_imp,
            "keep": mean_imp > 0.0,
        })

    return pd.DataFrame(rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)


def prune_noise_features(
    mda_df: pd.DataFrame,
    min_importance: float = 0.0,
    min_features: int = 4,
) -> List[str]:
    """Retiene únicamente las variables con importancia MDA OOS > min_importance."""
    if mda_df.empty:
        return []

    kept = mda_df[mda_df["mean_importance"] > min_importance]["feature"].tolist()
    if len(kept) < min_features:
        kept = mda_df.head(min(min_features, len(mda_df)))["feature"].tolist()
    return kept
