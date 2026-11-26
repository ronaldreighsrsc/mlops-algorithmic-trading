"""
Entrenador Institucional Meta-Labeling v3.0 (Tabular-First, Calibrado, Anti-Overfitting).

Pipeline estricto AFML (4 Capas Ortogonales):
  Capa 0: SymmetricCUSUMFilter (muestreo por llegada de información)
  Capa 1: BasePrimaryStrategy (lado económico causal side ∈ {-1, 0, +1})
  Capa 2: MetaBarrierLabeler + Sample Uniqueness + Features Multi-Escala + MDA OOS
          + Torneo Tabular (Logistic L2, XGBoost, LightGBM, RandomForest) con
            PurgedKFold([t0, t1]), optimización Log-Loss y PurgedPlattCalibrator
  Gobernanza: Cero OOS Threshold Snooping (umbral analítico de costo o validación interna),
              DSR empírico sobre N ensayos y CSCV PBO formal sobre matriz T x N.
  Despliegue: Exportación ONNX real autocontenida (Imputer + Scaler + Modelo + Platt)
              con paridad verificada < 1e-5 y contrato ModelBundle v3.0.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from src.evaluation.cpcv_auditor import CPCVAuditor
from src.evaluation.feature_importance import compute_purged_mda, prune_noise_features
from src.models.calibration import PurgedPlattCalibrator, compute_brier_and_ece
from src.models.onnx_exporter import export_sklearn_pipeline_to_onnx, update_onnx_manifest
from src.preprocessing.cusum_filter import SymmetricCUSUMFilter
from src.preprocessing.meta_labeling import MetaBarrierLabeler
from src.preprocessing.sample_weights import compute_sample_weights
from src.preprocessing.tabular_features import build_meta_feature_matrix
from src.strategies.primary_strategies import BasePrimaryStrategy, get_primary_strategy
from src.validation.purged_cv import PurgedKFold

logger = logging.getLogger("MetaTrainerV3")


def compute_analytical_probability_threshold(
    k_up: float,
    k_down: float,
    hurdle_multiplier: float = 2.5,
    friction_to_vol_ratio: float = 0.05,
) -> float:
    """
    Deriva analíticamente el punto de equilibrio económico de probabilidad mínima p_min
    a partir de las barreras (k_up, k_down) y la fricción esperada — eliminando el
    Test-Set Threshold Snooping (F-05).

    E[U] >= hurdle * friction  <=>  p * k_up - (1 - p) * k_down >= (hurdle + 1) * friction
    """
    cost_term = (1.0 + hurdle_multiplier) * friction_to_vol_ratio
    p_min = (k_down + cost_term) / max(k_up + k_down, 1e-6)
    return float(np.clip(p_min, 0.45, 0.85))


class MetaModelTrainerV3:
    """
    Entrenador completo de Meta-Labeling Tabular v3.0 con auditoría DSR/CSCV-PBO y ONNX real.
    """

    def __init__(
        self,
        symbol: str,
        primary_strategy: Optional[BasePrimaryStrategy] = None,
        cusum_kappa: float = 1.0,
        k_up: float = 2.0,
        k_down: float = 1.5,
        max_hold: int = 10,
        round_trip_cost: float = 0.0002,
        hurdle_multiplier: float = 2.5,
        n_splits: int = 5,
        pct_embargo: float = 0.01,
        decay_c: float = 0.5,
        enable_mda_pruning: bool = True,
        min_dsr_gate: float = 0.0,
        candidate_names: Optional[Tuple[str, ...]] = None,
        random_state: int = 42,
    ):
        self.symbol = symbol
        self.primary_strategy = primary_strategy or get_primary_strategy(
            "TrendPullbackPrimary", sma_slow=200, ema_fast=20, ema_slow=50, bilateral=True
        )
        self.cusum_kappa = float(cusum_kappa)
        self.k_up = float(k_up)
        self.k_down = float(k_down)
        self.max_hold = int(max_hold)
        self.round_trip_cost = float(round_trip_cost)
        self.hurdle_multiplier = float(hurdle_multiplier)
        self.n_splits = int(n_splits)
        self.pct_embargo = float(pct_embargo)
        self.decay_c = float(decay_c)
        self.enable_mda_pruning = bool(enable_mda_pruning)
        self.min_dsr_gate = float(min_dsr_gate)
        self.candidate_names = candidate_names
        self.random_state = int(random_state)

    def _build_candidate_factories(self, avg_uniqueness: float) -> Dict[str, Callable[[], Pipeline]]:
        """Familias tabulares regularizadas (Tabular-First para D1, cero BiLSTMs gigantes)."""
        rf_max_samples = float(np.clip(avg_uniqueness, 0.2, 0.95))
        rs = self.random_state

        all_factories = {
            "LOGISTIC_L2": lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=rs)),
            ]),
            "XGBOOST": lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", xgb.XGBClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.05,
                    min_child_weight=10,
                    subsample=0.75,
                    colsample_bytree=0.75,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    eval_metric="logloss",
                    random_state=rs,
                    n_jobs=1,
                )),
            ]),
            "LIGHTGBM": lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", lgb.LGBMClassifier(
                    n_estimators=120,
                    num_leaves=15,
                    max_depth=4,
                    learning_rate=0.05,
                    min_child_samples=20,
                    subsample=0.75,
                    colsample_bytree=0.75,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    random_state=rs,
                    verbose=-1,
                    n_jobs=1,
                )),
            ]),
            "RANDOM_FOREST": lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=150,
                    max_depth=5,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    max_samples=rf_max_samples,
                    random_state=rs,
                    n_jobs=1,
                )),
            ]),
        }
        if self.candidate_names:
            return {k: all_factories[k] for k in self.candidate_names if k in all_factories}
        return all_factories

    def prepare_meta_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Ejecuta Capa 0 (CUSUM) + Capa 1 (Modelo Primario) + Capa 2 (MetaBarrierLabeler,
        Features Tabulares y Sample Uniqueness Weights).
        Retorna: (events_df, X_events_df, sample_weights, avg_uniqueness).
        """
        cusum = SymmetricCUSUMFilter(kappa=self.cusum_kappa, vol_col="EGARCH_Vol")
        cusum_res = cusum.filter_events(df)

        sides = self.primary_strategy.generate_sides(df)

        # Serie causal de barras desde último evento CUSUM para contexto
        bars_since_series = np.zeros(len(df), dtype=np.float64)
        for pos, delta_bars in zip(cusum_res.event_positions, cusum_res.bars_since_last):
            bars_since_series[pos] = float(delta_bars)

        feat_matrix = build_meta_feature_matrix(
            df,
            primary_sides=sides,
            bars_since_cusum=bars_since_series,
        )

        labeler = MetaBarrierLabeler(
            k_up=self.k_up,
            k_down=self.k_down,
            max_hold=self.max_hold,
            round_trip_cost=self.round_trip_cost,
            require_tp_hit=False,
        )
        events_df = labeler.label_events(
            df,
            sides=sides,
            event_positions=cusum_res.event_positions if len(cusum_res.event_positions) > 0 else None,
        )
        if events_df.empty:
            return events_df, pd.DataFrame(), np.array([]), np.array([])

        X_events = feat_matrix.iloc[events_df["t0_pos"].values].reset_index(drop=True)
        events_df = events_df.reset_index(drop=True)

        weights, avg_u = compute_sample_weights(
            t0_pos=events_df["t0_pos"].values,
            t1_pos=events_df["t1_pos"].values,
            n_bars=len(df),
            close_prices=df["close"].astype(float).values,
            decay_c=self.decay_c,
            use_return_attribution=True,
        )
        return events_df, X_events, weights, avg_u

    def _fit_pipeline_weighted(self, pipe: Pipeline, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Pipeline:
        last_name = pipe.steps[-1][0]
        try:
            pipe.fit(X, y, **{f"{last_name}__sample_weight": w})
        except TypeError:
            pipe.fit(X, y)
        return pipe

    def train_and_audit(
        self,
        df: pd.DataFrame,
        results_dir: str,
        ffd_d_map: Optional[Dict[str, float]] = None,
        train_ratio: float = 0.75,
        export_onnx: bool = True,
    ) -> Dict[str, Any]:
        """
        Ejecuta entrenamiento completo sin leakage, poda MDA, calibración purgada Platt,
        auditoría DSR + CSCV PBO formal y exportación ONNX real verificada (< 1e-5).
        """
        events_df, X_events, weights, avg_u = self.prepare_meta_dataset(df)
        if len(events_df) < 60:
            raise ValueError(
                f"[{self.symbol}] Insuficientes eventos meta-etiquetados ({len(events_df)} < 60). "
                "Ajuste cusum_kappa o la historia de datos."
            )

        n_total = len(events_df)
        split_idx = int(n_total * train_ratio)

        # Purga estricta en la frontera Train / Holdout OOS (ningún evento train con t1 >= t0_oos)
        oos_start_t0 = events_df["t0"].iloc[split_idx]
        train_mask = (np.arange(n_total) < split_idx) & (events_df["t1"].values < oos_start_t0)
        oos_mask = np.arange(n_total) >= split_idx

        train_idx_all = np.where(train_mask)[0]
        oos_idx_all = np.where(oos_mask)[0]

        if len(train_idx_all) < 40 or len(oos_idx_all) < 15:
            raise ValueError(
                f"[{self.symbol}] Partición Train/OOS purgada insuficiente "
                f"(train={len(train_idx_all)}, oos={len(oos_idx_all)})."
            )

        X_tr_df = X_events.iloc[train_idx_all].reset_index(drop=True)
        y_tr = events_df["meta_label"].iloc[train_idx_all].values.astype(np.int32)
        w_tr = weights[train_idx_all]
        t0_tr = events_df["t0"].iloc[train_idx_all].reset_index(drop=True)
        t1_tr = events_df["t1"].iloc[train_idx_all].reset_index(drop=True)

        X_oos_df = X_events.iloc[oos_idx_all].reset_index(drop=True)
        y_oos = events_df["meta_label"].iloc[oos_idx_all].values.astype(np.int32)
        ret_oos = events_df["net_ret"].iloc[oos_idx_all].values.astype(np.float64)

        mean_u = float(np.mean(avg_u[train_idx_all]))
        selected_features = list(X_tr_df.columns)
        mda_df = pd.DataFrame()

        # 1. Poda de Variables por MDA Out-of-Sample Purgado (solo sobre Train!)
        if self.enable_mda_pruning and len(selected_features) > 6 and len(np.unique(y_tr)) >= 2:
            base_rf_factory = self._build_candidate_factories(mean_u)["RANDOM_FOREST"]
            mda_df = compute_purged_mda(
                pipeline_factory=base_rf_factory,
                X_df=X_tr_df,
                y=y_tr,
                t0=t0_tr,
                t1=t1_tr,
                sample_weights=w_tr,
                n_splits=min(self.n_splits, 4),
                pct_embargo=self.pct_embargo,
                n_repeats=2,
                random_state=self.random_state,
            )
            selected_features = prune_noise_features(mda_df, min_importance=0.0, min_features=min(6, len(X_tr_df.columns)))

        X_tr_sel = X_tr_df[selected_features].values.astype(np.float32)
        X_oos_sel = X_oos_df[selected_features].values.astype(np.float32)

        # Umbral económico analítico (cero snooping sobre OOS!)
        p_threshold = compute_analytical_probability_threshold(
            k_up=self.k_up,
            k_down=self.k_down,
            hurdle_multiplier=self.hurdle_multiplier,
        )

        # 2. Torneo Tabular con PurgedKFold + Calibración Platt sobre OOF purgado
        factories = self._build_candidate_factories(mean_u)
        cv = PurgedKFold(n_splits=min(self.n_splits, 5), pct_embargo=self.pct_embargo)

        candidate_results: Dict[str, Dict[str, Any]] = {}
        cscv_return_cols: List[np.ndarray] = []
        trial_sharpes_non_ann: List[float] = []

        for model_name, factory in factories.items():
            oof_raw_probs = np.full(len(y_tr), np.nan, dtype=np.float64)
            fold_losses = []

            for f_tr, f_val in cv.split(t0_tr, t1_tr):
                if len(f_tr) < 15 or len(f_val) < 3 or len(np.unique(y_tr[f_tr])) < 2:
                    continue
                pipe_cv = self._fit_pipeline_weighted(factory(), X_tr_sel[f_tr], y_tr[f_tr], w_tr[f_tr])
                p_val = np.clip(pipe_cv.predict_proba(X_tr_sel[f_val])[:, 1], 1e-6, 1.0 - 1e-6)
                oof_raw_probs[f_val] = p_val
                if len(np.unique(y_tr[f_val])) >= 2:
                    fold_losses.append(float(log_loss(y_tr[f_val], p_val, sample_weight=w_tr[f_val], labels=[0, 1])))

            valid_oof = ~np.isnan(oof_raw_probs)
            calibrator = PurgedPlattCalibrator()
            if np.sum(valid_oof) >= 15 and len(np.unique(y_tr[valid_oof])) >= 2:
                calibrator.fit(oof_raw_probs[valid_oof], y_tr[valid_oof], sample_weight=w_tr[valid_oof])

            # Entrenar pipeline maestro sobre todo el bloque Train purgado
            master_pipe = self._fit_pipeline_weighted(factory(), X_tr_sel, y_tr, w_tr)

            # Evaluar en Holdout OOS (tocado UNA sola vez con umbral analítico congelado)
            oos_raw = master_pipe.predict_proba(X_oos_sel)[:, 1]
            oos_cal = calibrator.transform(oos_raw)

            brier_oos, ece_oos = compute_brier_and_ece(y_oos, oos_cal)
            ll_oos = float(log_loss(y_oos, np.clip(oos_cal, 1e-6, 1.0 - 1e-6), labels=[0, 1])) if len(np.unique(y_oos)) >= 2 else brier_oos
            cv_ll = float(np.mean(fold_losses)) if fold_losses else ll_oos

            # Retornos realizados en OOS filtrados por p_threshold analítico
            taken_mask = oos_cal >= p_threshold
            strat_rets_oos = np.where(taken_mask, ret_oos, 0.0)

            # Serie completa (OOF + OOS) para matriz CSCV T x N
            full_probs = np.concatenate([
                calibrator.transform(np.where(valid_oof, oof_raw_probs, 0.5)),
                oos_cal,
            ])
            full_rets_raw = np.concatenate([
                events_df["net_ret"].iloc[train_idx_all].values.astype(np.float64),
                ret_oos,
            ])
            full_strat_rets = np.where(full_probs >= p_threshold, full_rets_raw, 0.0)
            cscv_return_cols.append(full_strat_rets)

            std_r = float(np.std(strat_rets_oos, ddof=1)) if len(strat_rets_oos) > 1 else 0.0
            sr_non_ann = float(np.mean(strat_rets_oos) / std_r) if std_r > 1e-12 else 0.0
            trial_sharpes_non_ann.append(sr_non_ann)

            candidate_results[model_name] = {
                "pipeline": master_pipe,
                "calibrator": calibrator,
                "cv_log_loss": cv_ll,
                "oos_log_loss": ll_oos,
                "brier_oos": brier_oos,
                "ece_oos": ece_oos,
                "oos_taken_trades": int(np.sum(taken_mask)),
                "oos_total_events": int(len(oos_cal)),
                "strat_rets_oos": strat_rets_oos,
                "oos_cal_probs": oos_cal,
            }

        # 3. Auditoría Estadística Conjunta: DSR Empírico + CSCV PBO Formal (T x N)
        auditor = CPCVAuditor(symbol=self.symbol, results_dir=results_dir)
        cscv_matrix = np.column_stack(cscv_return_cols)
        n_blocks = 8 if len(cscv_matrix) >= 80 else 4
        cscv_res = auditor.audit_cscv_pbo(cscv_matrix, n_blocks=n_blocks, purge_bars=min(self.max_hold, 5))

        for name, res in candidate_results.items():
            dsr_info = auditor.compute_deflated_sharpe_ratio(
                returns=res["strat_rets_oos"],
                n_trials=len(factories),
                trial_sharpes=np.asarray(trial_sharpes_non_ann, dtype=np.float64),
            )
            res["dsr_info"] = dsr_info

        # Seleccionar ganador por MENOR Log-Loss en Purged CV (sin mirar el OOS!)
        winner_name = min(candidate_results.keys(), key=lambda k: candidate_results[k]["cv_log_loss"])
        winner = candidate_results[winner_name]
        winner_dsr = float(winner["dsr_info"]["dsr"])
        passed_dsr_gate = winner_dsr >= self.min_dsr_gate

        # 4. Persistencia de Bundle v3.0 y Exportación ONNX Real Verificada (< 1e-5)
        models_dir = os.path.join(results_dir, "saved_models")
        os.makedirs(models_dir, exist_ok=True)

        bundle_filename = f"meta_{winner_name.lower()}_{self.symbol}_v3.pkl"
        bundle_path = os.path.join(models_dir, bundle_filename)
        onnx_filename = f"meta_{winner_name.lower()}_{self.symbol}_v3.onnx"
        onnx_path = os.path.join(models_dir, onnx_filename)

        X_parity = X_oos_sel[: min(200, len(X_oos_sel))]
        joblib.dump(
            {
                "pipeline": winner["pipeline"],
                "calibrator": winner["calibrator"].to_dict(),
                "feature_order": selected_features,
                "X_parity_sample": X_parity,
            },
            bundle_path,
        )

        sha256 = ""
        max_diff = 0.0
        if export_onnx:
            _, sha256, max_diff = export_sklearn_pipeline_to_onnx(
                pipeline=winner["pipeline"],
                output_path=onnx_path,
                X_parity_sample=X_parity,
                calibrator=winner["calibrator"],
                max_parity_tol=1e-5,
            )
            update_onnx_manifest(models_dir)

        model_bundle_cfg = {
            "schema_version": "3.0",
            "activo": self.symbol,
            "timeframe": "D1",
            "primary_strategy": {
                "name": self.primary_strategy.name,
                "params": self.primary_strategy.get_params(),
            },
            "cusum_config": {
                "kappa": self.cusum_kappa,
                "vol_col": "EGARCH_Vol",
            },
            "ffd_d_map": ffd_d_map or {},
            "meta_model": {
                "model_type": f"{winner_name}_CALIBRATED",
                "bundle_pkl": bundle_filename,
                "onnx_file": onnx_filename,
                "sha256": sha256,
                "onnx_parity_max_diff": max_diff,
                "feature_order": selected_features,
                "includes_scaler_in_onnx": True,
                "calibration_method": "purged_platt_logit",
                "analytical_prob_threshold": round(p_threshold, 4),
                "cv_log_loss": round(float(winner["cv_log_loss"]), 5),
                "oos_log_loss": round(float(winner["oos_log_loss"]), 5),
                "brier_score_oos": round(float(winner["brier_oos"]), 5),
                "ece_oos": round(float(winner["ece_oos"]), 5),
                "dsr_oos": round(winner_dsr, 5),
                "passed_dsr_gate": bool(passed_dsr_gate),
                "pbo_cscv": round(float(cscv_res["pbo"]), 5),
                "avg_sample_uniqueness": round(mean_u, 4),
            },
            "barriers": {
                "k_up": self.k_up,
                "k_down": self.k_down,
                "max_hold": self.max_hold,
            },
            "risk": {
                "optimal_risk_pct": 0.015,
                "kill_switch_mdd": -0.20,
                "hurdle_multiplier": self.hurdle_multiplier,
            },
        }

        cfg_path = os.path.join(results_dir, f"campeon_{self.symbol}_v3.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(model_bundle_cfg, f, indent=4)

        return {
            "winner": winner_name,
            "config_path": cfg_path,
            "onnx_path": onnx_path,
            "model_bundle": model_bundle_cfg,
            "cscv": cscv_res,
            "mda": mda_df,
            "candidates_summary": {
                k: {
                    "cv_log_loss": v["cv_log_loss"],
                    "oos_log_loss": v["oos_log_loss"],
                    "brier_oos": v["brier_oos"],
                    "ece_oos": v["ece_oos"],
                    "dsr": v["dsr_info"]["dsr"],
                    "oos_taken_trades": v["oos_taken_trades"],
                }
                for k, v in candidate_results.items()
            },
        }


def run_preregistered_v31_tournament(
    symbol: str,
    df: pd.DataFrame,
    primaries: List[BasePrimaryStrategy],
    results_dir: str,
    candidate_names: Tuple[str, ...] = ("LOGISTIC_L2", "RANDOM_FOREST"),
    cusum_kappa: float = 1.0,
    k_up: float = 2.0,
    k_down: float = 1.5,
    max_hold: int = 10,
    round_trip_cost: float = 0.0002,
    hurdle_multiplier: float = 2.5,
    train_ratio: float = 0.70,
    enable_mda_pruning: bool = True,
    ffd_d_map: Optional[Dict[str, float]] = None,
    export_onnx: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Protocolo Experimental Cerrado v3.1 (SPEC_QUANT_V3_1_ALPHA_SIGNALS §3):
    - Evalúa un presupuesto pre-registrado fijo de P primarias × M modelos tabulares
      (por defecto 3 x 2 = 6 ensayos totales).
    - Usa un corte cronológico fijo `train_ratio=0.70` compartido entre todas las
      primarias para que el Holdout OOS (30%) y la matriz CSCV T x N estén exactamente
      alineados en el tiempo.
    - Selecciona al ganador por MENOR CV Log-Loss purgado en Desarrollo (cero snooping OOS).
    - Evalúa 5 compuertas institucionales obligatorias antes de aprobar despliegue real.
    """
    if len(df) < 250:
        raise ValueError(f"[{symbol}] Historia insuficiente para torneo v3.1 ({len(df)} barras).")

    split_bar_idx = int(len(df) * train_ratio)
    time_col_vals = pd.to_datetime(df["time"]).values if "time" in df.columns else df.index.values
    holdout_start_time = time_col_vals[split_bar_idx]

    p_threshold = compute_analytical_probability_threshold(
        k_up=k_up,
        k_down=k_down,
        hurdle_multiplier=hurdle_multiplier,
    )

    n_bars = len(df)
    bar_aligned_return_cols: List[np.ndarray] = []
    holdout_sharpes_non_ann: List[float] = []
    trial_records: List[Dict[str, Any]] = []

    for prim in primaries:
        trainer = MetaModelTrainerV3(
            symbol=symbol,
            primary_strategy=prim,
            cusum_kappa=cusum_kappa,
            k_up=k_up,
            k_down=k_down,
            max_hold=max_hold,
            round_trip_cost=round_trip_cost,
            hurdle_multiplier=hurdle_multiplier,
            n_splits=5,
            pct_embargo=0.01,
            enable_mda_pruning=enable_mda_pruning,
            candidate_names=candidate_names,
            random_state=random_state,
        )
        events_df, X_events, weights, avg_u = trainer.prepare_meta_dataset(df)
        if events_df.empty:
            continue

        t0_vals = pd.to_datetime(events_df["t0"]).values
        t1_vals = pd.to_datetime(events_df["t1"]).values
        train_mask = (t0_vals < holdout_start_time) & (t1_vals < holdout_start_time)
        oos_mask = t0_vals >= holdout_start_time

        train_idx = np.where(train_mask)[0]
        oos_idx = np.where(oos_mask)[0]
        if len(train_idx) < 35 or len(oos_idx) < 10:
            logger.warning(
                f"[{symbol} | {prim.name}] Omitida por insuficiencia de eventos "
                f"(train={len(train_idx)}, oos={len(oos_idx)})."
            )
            continue

        X_tr_df = X_events.iloc[train_idx].reset_index(drop=True)
        y_tr = events_df["meta_label"].iloc[train_idx].values.astype(np.int32)
        w_tr = weights[train_idx]
        t0_tr = events_df["t0"].iloc[train_idx].reset_index(drop=True)
        t1_tr = events_df["t1"].iloc[train_idx].reset_index(drop=True)

        X_oos_df = X_events.iloc[oos_idx].reset_index(drop=True)
        y_oos = events_df["meta_label"].iloc[oos_idx].values.astype(np.int32)
        ret_oos = events_df["net_ret"].iloc[oos_idx].values.astype(np.float64)

        mean_u = float(np.mean(avg_u[train_idx]))
        selected_features = list(X_tr_df.columns)
        if enable_mda_pruning and len(selected_features) > 6 and len(np.unique(y_tr)) >= 2:
            rf_factory = trainer._build_candidate_factories(mean_u)["RANDOM_FOREST"]
            mda_df = compute_purged_mda(
                pipeline_factory=rf_factory,
                X_df=X_tr_df,
                y=y_tr,
                t0=t0_tr,
                t1=t1_tr,
                sample_weights=w_tr,
                n_splits=4,
                pct_embargo=0.01,
                n_repeats=2,
                random_state=random_state,
            )
            selected_features = prune_noise_features(mda_df, min_importance=0.0, min_features=min(6, len(X_tr_df.columns)))

        X_tr_sel = X_tr_df[selected_features].values.astype(np.float32)
        X_oos_sel = X_oos_df[selected_features].values.astype(np.float32)

        cv = PurgedKFold(n_splits=min(5, max(2, len(train_idx) // 15)), pct_embargo=0.01)
        factories = trainer._build_candidate_factories(mean_u)

        for model_name, factory in factories.items():
            oof_raw_probs = np.full(len(y_tr), np.nan, dtype=np.float64)
            fold_losses = []

            for f_tr, f_val in cv.split(t0_tr, t1_tr):
                if len(f_tr) < 15 or len(f_val) < 3 or len(np.unique(y_tr[f_tr])) < 2:
                    continue
                pipe_cv = trainer._fit_pipeline_weighted(factory(), X_tr_sel[f_tr], y_tr[f_tr], w_tr[f_tr])
                p_val = np.clip(pipe_cv.predict_proba(X_tr_sel[f_val])[:, 1], 1e-6, 1.0 - 1e-6)
                oof_raw_probs[f_val] = p_val
                if len(np.unique(y_tr[f_val])) >= 2:
                    fold_losses.append(float(log_loss(y_tr[f_val], p_val, sample_weight=w_tr[f_val], labels=[0, 1])))

            valid_oof = ~np.isnan(oof_raw_probs)
            calibrator = PurgedPlattCalibrator()
            if np.sum(valid_oof) >= 15 and len(np.unique(y_tr[valid_oof])) >= 2:
                calibrator.fit(oof_raw_probs[valid_oof], y_tr[valid_oof], sample_weight=w_tr[valid_oof])

            master_pipe = trainer._fit_pipeline_weighted(factory(), X_tr_sel, y_tr, w_tr)
            oos_cal = calibrator.transform(master_pipe.predict_proba(X_oos_sel)[:, 1])

            brier_oos, ece_oos = compute_brier_and_ece(y_oos, oos_cal)
            ll_oos = float(log_loss(y_oos, np.clip(oos_cal, 1e-6, 1.0 - 1e-6), labels=[0, 1])) if len(np.unique(y_oos)) >= 2 else brier_oos
            cv_ll = float(np.mean(fold_losses)) if fold_losses else ll_oos

            # Construir vector de retornos alineado en la grilla exacta de T barras de df
            bar_rets = np.zeros(n_bars, dtype=np.float64)
            tr_probs = calibrator.transform(np.where(valid_oof, oof_raw_probs, 0.5))
            for pos_in_tr, idx_ev in enumerate(train_idx):
                if tr_probs[pos_in_tr] >= p_threshold:
                    exit_bar = int(events_df["t1_pos"].iloc[idx_ev])
                    if 0 <= exit_bar < n_bars:
                        bar_rets[exit_bar] += float(events_df["net_ret"].iloc[idx_ev])

            taken_oos = oos_cal >= p_threshold
            for pos_in_oos, idx_ev in enumerate(oos_idx):
                if taken_oos[pos_in_oos]:
                    exit_bar = int(events_df["t1_pos"].iloc[idx_ev])
                    if 0 <= exit_bar < n_bars:
                        bar_rets[exit_bar] += float(events_df["net_ret"].iloc[idx_ev])

            holdout_bar_rets = bar_rets[split_bar_idx:]
            std_h = float(np.std(holdout_bar_rets, ddof=1)) if len(holdout_bar_rets) > 1 else 0.0
            mean_h = float(np.mean(holdout_bar_rets))
            sr_h_non_ann = (mean_h / std_h) if std_h > 1e-12 else 0.0
            sr_h_ann = sr_h_non_ann * float(np.sqrt(252.0))

            bar_aligned_return_cols.append(bar_rets)
            holdout_sharpes_non_ann.append(sr_h_non_ann)

            trial_records.append({
                "trial_id": f"{prim.name}__{model_name}",
                "primary": prim,
                "primary_name": prim.name,
                "model_name": model_name,
                "pipeline": master_pipe,
                "calibrator": calibrator,
                "selected_features": selected_features,
                "X_parity_sample": X_oos_sel[: min(200, len(X_oos_sel))],
                "cv_log_loss": cv_ll,
                "oos_log_loss": ll_oos,
                "brier_oos": brier_oos,
                "ece_oos": ece_oos,
                "oos_taken_trades": int(np.sum(taken_oos)),
                "oos_total_events": int(len(oos_cal)),
                "oos_net_sharpe": sr_h_ann,
                "holdout_bar_rets": holdout_bar_rets,
                "avg_uniqueness": mean_u,
            })

    if not trial_records:
        raise ValueError(f"[{symbol}] Ningún ensayo del torneo v3.1 tuvo eventos suficientes.")

    # Benchmark pasivo / SMA-200 sobre el MISMO bloque Holdout [split_bar_idx:]
    closes = df["close"].astype(float).values
    holdout_mkt_rets = np.diff(closes[split_bar_idx - 1 :]) / np.maximum(closes[split_bar_idx - 1 : -1], 1e-12)
    sma200 = pd.Series(closes).rolling(200, min_periods=50).mean().values
    # Regla causal Long-Cash en Holdout: close[t-1] > sma200[t-1]
    sma_sig = np.where(
        (~np.isnan(sma200[split_bar_idx - 1 : -1])) & (closes[split_bar_idx - 1 : -1] > sma200[split_bar_idx - 1 : -1]),
        1.0,
        0.0,
    )
    holdout_sma_rets = holdout_mkt_rets * sma_sig

    def _ann_sharpe(arr: np.ndarray) -> float:
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return float((np.mean(arr) / s) * np.sqrt(252.0)) if s > 1e-12 else 0.0

    bench_bh_sharpe = _ann_sharpe(holdout_mkt_rets)
    bench_sma_sharpe = _ann_sharpe(holdout_sma_rets)
    benchmark_sharpe = max(bench_bh_sharpe, bench_sma_sharpe)

    # CSCV PBO formal sobre matriz T x N alineada entre todos los ensayos
    auditor = CPCVAuditor(symbol=f"{symbol}_v31", results_dir=results_dir)
    if len(bar_aligned_return_cols) >= 2:
        cscv_matrix = np.column_stack(bar_aligned_return_cols)
        cscv_res = auditor.audit_cscv_pbo(cscv_matrix, n_blocks=8, purge_bars=min(max_hold, 5))
    else:
        cscv_res = {"pbo": 1.0, "n_combinations": 0, "n_strategies": len(bar_aligned_return_cols)}

    # Seleccionar ganador estrictamente por MENOR CV Log-Loss dentro del 70% Desarrollo
    winner = min(trial_records, key=lambda r: r["cv_log_loss"])
    dsr_info = auditor.compute_deflated_sharpe_ratio(
        returns=winner["holdout_bar_rets"],
        n_trials=len(trial_records),
        trial_sharpes=np.asarray(holdout_sharpes_non_ann, dtype=np.float64),
    )

    # 5 Compuertas Institucionales de Aceptación (SPEC v3.1 §3)
    gates = {
        "oos_log_loss_lt_067": bool(winner["oos_log_loss"] < 0.67),
        "oos_ece_lt_005": bool(winner["ece_oos"] < 0.05),
        "dsr_ge_095": bool(dsr_info["dsr"] >= 0.95),
        "cscv_pbo_lt_020": bool(cscv_res["pbo"] < 0.20),
        "sharpe_gt_benchmark": bool(winner["oos_net_sharpe"] > benchmark_sharpe),
    }
    gate_approved = all(gates.values())

    # Exportación ONNX real verificada (< 1e-5)
    models_dir = os.path.join(results_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    bundle_filename = f"meta_{winner['model_name'].lower()}_{symbol}_v31.pkl"
    onnx_filename = f"meta_{winner['model_name'].lower()}_{symbol}_v31.onnx"
    bundle_path = os.path.join(models_dir, bundle_filename)
    onnx_path = os.path.join(models_dir, onnx_filename)

    joblib.dump(
        {
            "pipeline": winner["pipeline"],
            "calibrator": winner["calibrator"].to_dict(),
            "feature_order": winner["selected_features"],
            "X_parity_sample": winner["X_parity_sample"],
        },
        bundle_path,
    )

    sha256, max_diff = "", 0.0
    if export_onnx and len(winner["X_parity_sample"]) > 0:
        _, sha256, max_diff = export_sklearn_pipeline_to_onnx(
            pipeline=winner["pipeline"],
            output_path=onnx_path,
            X_parity_sample=winner["X_parity_sample"],
            calibrator=winner["calibrator"],
            max_parity_tol=1e-5,
        )
        update_onnx_manifest(models_dir)

    bundle_cfg = {
        "schema_version": "3.1",
        "activo": symbol,
        "timeframe": "D1",
        "primary_strategy": {
            "name": winner["primary_name"],
            "params": winner["primary"].get_params(),
        },
        "cusum_config": {"kappa": cusum_kappa, "vol_col": "EGARCH_Vol"},
        "ffd_d_map": ffd_d_map or {},
        "meta_model": {
            "model_type": f"{winner['model_name']}_CALIBRATED",
            "bundle_pkl": bundle_filename,
            "onnx_file": onnx_filename,
            "sha256": sha256,
            "onnx_parity_max_diff": max_diff,
            "feature_order": winner["selected_features"],
            "includes_scaler_in_onnx": True,
            "calibration_method": "purged_platt_logit",
            "analytical_prob_threshold": round(p_threshold, 4),
            "n_trials_preregistered": len(trial_records),
            "cv_log_loss": round(winner["cv_log_loss"], 5),
            "oos_log_loss": round(winner["oos_log_loss"], 5),
            "brier_score_oos": round(winner["brier_oos"], 5),
            "ece_oos": round(winner["ece_oos"], 5),
            "oos_net_sharpe": round(winner["oos_net_sharpe"], 4),
            "benchmark_sharpe": round(benchmark_sharpe, 4),
            "dsr_oos": round(float(dsr_info["dsr"]), 5),
            "pbo_cscv": round(float(cscv_res["pbo"]), 5),
            "gates": gates,
            "gate_approved": gate_approved,
            "avg_sample_uniqueness": round(winner["avg_uniqueness"], 4),
        },
        "barriers": {"k_up": k_up, "k_down": k_down, "max_hold": max_hold},
        "risk": {"optimal_risk_pct": 0.015, "kill_switch_mdd": -0.20, "hurdle_multiplier": hurdle_multiplier},
    }

    cfg_path = os.path.join(results_dir, f"campeon_{symbol}_v31.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(bundle_cfg, f, indent=4)

    return {
        "symbol": symbol,
        "winner_trial": winner["trial_id"],
        "gate_approved": gate_approved,
        "gates": gates,
        "model_bundle": bundle_cfg,
        "cscv": cscv_res,
        "dsr": dsr_info,
        "trials_table": [
            {
                "trial_id": r["trial_id"],
                "n_features": len(r["selected_features"]),
                "cv_log_loss": round(r["cv_log_loss"], 5),
                "oos_log_loss": round(r["oos_log_loss"], 5),
                "brier_oos": round(r["brier_oos"], 5),
                "ece_oos": round(r["ece_oos"], 5),
                "oos_taken_trades": r["oos_taken_trades"],
                "oos_total_events": r["oos_total_events"],
                "oos_net_sharpe": round(r["oos_net_sharpe"], 3),
            }
            for r in trial_records
        ],
    }
