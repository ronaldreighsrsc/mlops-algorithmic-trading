"""
Clasificador de Regímenes de Mercado Dinámicos Institucionales (v2.0).
Combina:
1. Gaussian HMM de 3 Estados (Bull Trend, Bear Trend, Choppy/Turbulento) con Modulación de Kelly.
2. Test no paramétrico de Kolmogorov-Smirnov (KS-Test) sobre retornos móviles contra in-sample.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Códigos de Régimen
REGIME_BULL = 0      # Tendencia Alcista (Kelly 1.0x, sesgo Long)
REGIME_BEAR = 1      # Tendencia Bajista (Kelly 1.0x, sesgo Short)
REGIME_CHOPPY = 2    # Rango Turbulento/Lateral (Kelly 0.25x o Cash)

REGIME_NAMES = {
    REGIME_BULL: "BULL_TREND",
    REGIME_BEAR: "BEAR_TREND",
    REGIME_CHOPPY: "CHOPPY_HIGH_VOL",
}


@dataclass
class RegimeDiagnosis:
    regime_state: int
    regime_name: str
    regime_kelly_multiplier: float
    ks_drift_detected: bool
    ks_p_value: float
    ks_safety_multiplier: float
    effective_kelly_factor: float
    summary: str


class InstitutionalRegimeDetector:
    """
    Detector de Regímenes HMM de 3 Estados y Monitor de Deriva Kolmogorov-Smirnov.
    """

    def __init__(self, n_components: int = 3, random_state: int = 42, ks_alpha: float = 0.01):
        self.n_components = n_components
        self.random_state = random_state
        self.ks_alpha = ks_alpha

        self.model = None
        self.scaler = StandardScaler()
        self.state_mapping: Dict[int, int] = {} # raw_state -> semantic_state
        self.reference_returns: Optional[np.ndarray] = None # in-sample baseline for KS-test

    def fit(self, X_features: np.ndarray, returns_col_idx: int = 0):
        """
        Entrena el Gaussian HMM sobre features estandarizadas (ej. FFD returns, EGARCH Vol, Volume).
        Mapea semánticamente los estados a BULL (0), BEAR (1) y CHOPPY (2).
        """
        from hmmlearn import hmm

        X_arr = np.asarray(X_features, dtype=np.float64)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(-1, 1)

        # Guardar retornos in-sample de referencia para el test KS
        self.reference_returns = X_arr[:, returns_col_idx].copy()

        X_scaled = self.scaler.fit_transform(X_arr)
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            random_state=self.random_state,
            n_iter=150,
        )
        self.model.fit(X_scaled)

        # Mapeo semántico de estados:
        # Calculamos la varianza total de cada estado (traza de la covarianza)
        # y la media del retorno en cada estado.
        state_variances = []
        state_means = []

        for i in range(self.n_components):
            cov_mat = self.model.covars_[i]
            total_var = float(np.trace(cov_mat))
            mean_ret = float(self.model.means_[i][returns_col_idx])
            state_variances.append(total_var)
            state_means.append(mean_ret)

        # 1. El estado con mayor varianza relativa o retorno cercano a 0 con alta volatilidad es CHOPPY
        choppy_raw = int(np.argmax(state_variances))

        # 2. Entre los dos restantes, el de mayor retorno medio es BULL y el menor es BEAR
        remaining = [s for s in range(self.n_components) if s != choppy_raw]
        if len(remaining) == 2:
            if state_means[remaining[0]] >= state_means[remaining[1]]:
                bull_raw = remaining[0]
                bear_raw = remaining[1]
            else:
                bull_raw = remaining[1]
                bear_raw = remaining[0]
        else:
            bull_raw = 0
            bear_raw = 1

        self.state_mapping = {
            bull_raw: REGIME_BULL,
            bear_raw: REGIME_BEAR,
            choppy_raw: REGIME_CHOPPY,
        }

    def predict_regime(self, X_recent: np.ndarray) -> int:
        """Retorna el estado semántico actual (0: Bull, 1: Bear, 2: Choppy)."""
        if self.model is None:
            return REGIME_BULL

        X_arr = np.asarray(X_recent, dtype=np.float64)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)

        X_scaled = self.scaler.transform(X_arr)
        raw_state = int(self.model.predict(X_scaled)[-1])
        return self.state_mapping.get(raw_state, REGIME_BULL)

    def evaluate_ks_drift(self, recent_returns: np.ndarray) -> Tuple[bool, float, float]:
        """
        Ejecuta el test no paramétrico de Kolmogorov-Smirnov (KS-Test)
        sobre los retornos recientes (ej. últimas 50 velas) contra el histórico in-sample.
        Retorna: (is_drift, p_value, safety_multiplier)
        """
        if self.reference_returns is None or len(self.reference_returns) < 30:
            return False, 1.0, 1.0

        r_recent = np.asarray(recent_returns, dtype=np.float64).flatten()
        r_recent = r_recent[~np.isnan(r_recent)]

        if len(r_recent) < 20:
            return False, 1.0, 1.0

        # Test de Kolmogorov-Smirnov a dos muestras
        res = stats.ks_2samp(r_recent, self.reference_returns)
        p_val = float(res.pvalue)

        is_drift = bool(p_val < self.ks_alpha)
        # Si hay drift estadístico significativo, reducir Kelly al 50%
        safety_multiplier = 0.50 if is_drift else 1.0

        return is_drift, p_val, safety_multiplier

    def diagnose(self, X_recent_features: np.ndarray, recent_returns: np.ndarray) -> RegimeDiagnosis:
        """
        Diagnóstico integral del régimen y estado de deriva.
        """
        regime_state = self.predict_regime(X_recent_features)
        regime_name = REGIME_NAMES.get(regime_state, "UNKNOWN")

        # Multiplicador por estado HMM
        if regime_state == REGIME_CHOPPY:
            regime_kelly = 0.25
        else:
            regime_kelly = 1.0

        # Deriva Kolmogorov-Smirnov
        is_drift, ks_p_val, ks_safety = self.evaluate_ks_drift(recent_returns)

        # Factor de Kelly efectivo acumulado
        effective_factor = regime_kelly * ks_safety

        summary = (
            f"Regime: {regime_name} (Kelly {regime_kelly:.2f}x) | "
            f"KS-Drift: {'SÍ (p=' + f'{ks_p_val:.4f}' + ')' if is_drift else 'NO (p=' + f'{ks_p_val:.4f}' + ')'} "
            f"(Safety {ks_safety:.2f}x) -> Kelly Efectivo: {effective_factor:.2f}x"
        )

        return RegimeDiagnosis(
            regime_state=regime_state,
            regime_name=regime_name,
            regime_kelly_multiplier=regime_kelly,
            ks_drift_detected=is_drift,
            ks_p_value=round(ks_p_val, 5),
            ks_safety_multiplier=ks_safety,
            effective_kelly_factor=round(effective_factor, 4),
            summary=summary,
        )

    def save(self, filepath: str):
        """Serializa el detector de régimen institucional."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "state_mapping": self.state_mapping,
            "reference_returns": self.reference_returns,
            "ks_alpha": self.ks_alpha,
            "n_components": self.n_components,
            "random_state": self.random_state,
        }, filepath)

    def load(self, filepath: str):
        """Carga el detector desde disco."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.state_mapping = data["state_mapping"]
        self.reference_returns = data.get("reference_returns", None)
        self.ks_alpha = data.get("ks_alpha", 0.01)
        self.n_components = data.get("n_components", 3)
        self.random_state = data.get("random_state", 42)
