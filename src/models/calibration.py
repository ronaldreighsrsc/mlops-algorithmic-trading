"""
Calibrador Probabilístico Causal (Platt Scaling en Log-Odds) para Meta-Labeling v3.0.

A diferencia de sklearn.calibration.CalibratedClassifierCV(cv=3) —que usa StratifiedKFold
no purgado (fuga temporal) y presenta divergencias en skl2onnx al envolver XGBoost—,
este calibrador se ajusta estrictamente sobre predicciones out-of-fold purgadas y se
exporta como nodos ONNX nativos con paridad numérica exacta (< 1e-6).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _clip_and_logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class PurgedPlattCalibrator:
    """
    Calibración Platt sobre log-odds:
        p_cal = sigmoid(coef * logit(clip(p_raw, eps, 1-eps)) + intercept)
    """

    def __init__(self, eps: float = 1e-6, c_reg: float = 10.0):
        self.eps = float(eps)
        self.c_reg = float(c_reg)
        self.coef_: float = 1.0
        self.intercept_: float = 0.0
        self.is_fitted_: bool = False

    def fit(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "PurgedPlattCalibrator":
        """Ajusta (coef_, intercept_) usando predicciones out-of-fold purgadas."""
        y_arr = np.asarray(y_true, dtype=np.int32).ravel()
        p_arr = np.asarray(raw_probs, dtype=np.float64).ravel()

        if len(p_arr) < 10 or len(np.unique(y_arr)) < 2:
            # Fallback identidad si no hay ambas clases en calibración
            self.coef_ = 1.0
            self.intercept_ = 0.0
            self.is_fitted_ = True
            return self

        logits = _clip_and_logit(p_arr, self.eps).reshape(-1, 1)
        lr = LogisticRegression(C=self.c_reg, solver="lbfgs", max_iter=500)
        lr.fit(logits, y_arr, sample_weight=sample_weight)

        # Mantener monotonicidad positiva (probabilidad mayor => calibrada mayor)
        coef_val = float(lr.coef_[0, 0])
        if coef_val <= 0.0:
            self.coef_ = 1.0
            self.intercept_ = 0.0
        else:
            self.coef_ = coef_val
            self.intercept_ = float(lr.intercept_[0])

        self.is_fitted_ = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transforma probabilidades crudas en probabilidades calibradas [0, 1]."""
        logits = _clip_and_logit(raw_probs, self.eps)
        z = np.clip(self.coef_ * logits + self.intercept_, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict:
        return {
            "eps": self.eps,
            "coef": self.coef_,
            "intercept": self.intercept_,
            "is_fitted": self.is_fitted_,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PurgedPlattCalibrator":
        inst = cls(eps=data.get("eps", 1e-6))
        inst.coef_ = float(data.get("coef", 1.0))
        inst.intercept_ = float(data.get("intercept", 0.0))
        inst.is_fitted_ = bool(data.get("is_fitted", True))
        return inst


def compute_brier_and_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """Calcula Brier Score y Expected Calibration Error (ECE)."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)
    if len(y_t) == 0:
        return 1.0, 1.0

    brier = float(np.mean((y_p - y_t) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_t)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (y_p >= lo) & (y_p <= hi)
        else:
            mask = (y_p >= lo) & (y_p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            avg_conf = float(np.mean(y_p[mask]))
            avg_acc = float(np.mean(y_t[mask]))
            ece += (cnt / n) * abs(avg_acc - avg_conf)

    return brier, float(ece)
